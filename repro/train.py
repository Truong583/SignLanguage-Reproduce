"""Single-node DDP, exact effective-batch arithmetic, epoch-boundary resume."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time
import signal

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler, Subset
import yaml

from .data import SignDataset, collate, read_manifest, build_vocab
from .model import SignModel
from .metrics import ctc_decode, corpus_wer, translation_scores
from .runtime import batch_plan, seed_everything, rng_state, restore_rng, atomic_save, sha256, environment, write_json
from .persistence import make_store, publish


class GlobalBatchSampler(Sampler):
    def __init__(self,size,global_batch,rank,world,seed,epoch,skip_updates=0):
        ids = torch.randperm(size,generator=torch.Generator().manual_seed(seed+epoch)).tolist()
        usable = size//global_batch*global_batch
        if not usable: raise ValueError('Training split is smaller than effective global batch')
        if not 0 <= skip_updates <= usable//global_batch: raise ValueError('Invalid resume cursor')
        self.ids = ids[skip_updates*global_batch:usable][rank::world]
    def __iter__(self): return iter(self.ids)
    def __len__(self): return len(self.ids)


def gather_object(value, world):
    if world==1: return [value]
    results = [None]*world
    dist.all_gather_object(results,value)
    return results


def evaluate(model,cfg,vocab,split,device,rank,world,amp):
    dataset = SignDataset(cfg[split],cfg['data_root'],input_kind=cfg.get('input_kind','rgb'))
    # No duplicate padding examples and no DDP forward/buffer collectives in eval.
    subset = Subset(dataset,list(range(rank,len(dataset),world)))
    loader = DataLoader(subset,batch_size=cfg.get('eval_batch',1),num_workers=cfg.get('workers',2),
                        collate_fn=collate,pin_memory=device.type=='cuda')
    model.eval()
    predictions=[]
    with torch.no_grad():
        for batch in loader:
            with amp():
                out = model.features(batch['video'].to(device,non_blocking=True),batch['lengths'])
                texts = model.generate(out)
            conv = out['conv'].float().log_softmax(-1).cpu().numpy()
            sequence = out['sequence'].float().log_softmax(-1).cpu().numpy()
            for i,(row,length) in enumerate(zip(batch['rows'],out['lengths'].tolist())):
                result={'id':row['id'],'reference':row[cfg.get('target_field','gloss')],
                        'text_reference':row.get('text',''),'text_prediction':texts[i]}
                for head,logp in [('conv',conv),('sequence',sequence)]:
                    result[head]=' '.join(vocab[t] for t in ctc_decode(logp[:length,i],cfg.get('ctc_beam',10)))
                predictions.append(result)
    predictions = sorted([p for group in gather_object(predictions,world) for p in group],key=lambda p:p['id'])
    if len(predictions)!=len(dataset) or len({r['id'] for r in predictions})!=len(dataset):
        raise RuntimeError('Evaluation coverage mismatch')
    scores = {head:corpus_wer([r['reference'] for r in predictions],[r[head] for r in predictions])
              for head in ('conv','sequence')}
    if model.translation is not None:
        scores.update(translation_scores([r['text_reference'] for r in predictions],
                                         [r['text_prediction'] for r in predictions],cfg.get('text_level','word')))
    scores['samples']=len(predictions)
    return scores,predictions


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--mode',choices=['train','eval'],default='train')
    parser.add_argument('--split',choices=['dev','test'],default='test')
    parser.add_argument('--checkpoint')
    parser.add_argument('--resume')
    parser.add_argument('--initialize-from')
    parser.add_argument('--micro-batch',type=int)
    parser.add_argument('--workers',type=int)
    args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    if args.micro_batch: cfg['micro_batch']=args.micro_batch
    if args.workers is not None: cfg['workers']=args.workers
    if args.resume and args.initialize_from: parser.error('Choose resume OR initialize-from')
    rank,world,local = (int(os.environ.get(k,default)) for k,default in [('RANK','0'),('WORLD_SIZE','1'),('LOCAL_RANK','0')])
    use_cuda=torch.cuda.is_available() and cfg.get('device','cuda')!='cpu'
    if not use_cuda: torch.backends.cudnn.enabled=False
    device=torch.device('cuda',local) if use_cuda else torch.device('cpu')
    if use_cuda: torch.cuda.set_device(device)
    if world>1: dist.init_process_group('nccl' if use_cuda else 'gloo')
    torch.set_num_threads(cfg.get('cpu_threads',2))
    seed=cfg.get('seed',0)
    seed_everything(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    precision=cfg.get('precision','fp16') if use_cuda else 'fp32'
    if precision=='bf16' and not torch.cuda.is_bf16_supported(): raise ValueError('GPU does not support BF16')
    amp=lambda:torch.autocast(device.type,dtype=torch.float16 if precision=='fp16' else torch.bfloat16,enabled=precision!='fp32')
    scaler=torch.amp.GradScaler('cuda',enabled=precision=='fp16')
    output=Path(cfg['output'])
    if rank==0: output.mkdir(parents=True,exist_ok=True)
    if world>1: dist.barrier()
    checkpoint_path=args.resume or args.checkpoint or args.initialize_from
    # Only own/trusted checkpoints. Optimizer and RNG require weights_only=False.
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False) if checkpoint_path else None
    manifests={split:sha256(cfg[split]) for split in ['train','dev','test']}
    vocab=build_vocab(read_manifest(cfg['train']),cfg.get('target_field','gloss'))
    if checkpoint:
        if checkpoint['vocab']!=vocab: raise ValueError('Vocabulary differs from checkpoint')
        if (args.resume or args.checkpoint) and checkpoint['manifests']!=manifests:
            raise ValueError('Manifest hashes differ from checkpoint')
    model=SignModel(cfg,vocab,initialize=checkpoint is None).to(device)
    if checkpoint:
        if args.initialize_from:
            result=model.load_state_dict(checkpoint['model'],strict=False)
            if result.unexpected_keys or any(not k.startswith(('translation.','mapping.')) for k in result.missing_keys):
                raise ValueError(f'Incompatible pretraining checkpoint: {result}')
        else:
            for key in ('input_kind','hidden_size','hsg','graph_order','graph_conv','task','target_field','language',
                        'paper_ablation','backbone','module_mask','module_order','graph_scales','local_k','temporal_k',
                        'distance','lsg_type','tsg_type','hsg_type','drop_edge'):
                if cfg.get(key)!=checkpoint['config'].get(key): raise ValueError(f'Checkpoint/config mismatch: {key}')
            model.load_state_dict(checkpoint['model'],strict=True)
    if args.mode=='eval':
        if not args.checkpoint: raise ValueError('Evaluation requires --checkpoint; no random model scoring')
        scores,predictions=evaluate(model,cfg,vocab,args.split,device,rank,world,amp)
        if rank==0:
            write_json(output/f'{args.split}_metrics.json',scores)
            write_json(output/f'{args.split}_predictions.json',predictions)
            print(json.dumps(scores),flush=True)
        if world>1: dist.destroy_process_group()
        return
    if args.checkpoint: raise ValueError('Use --resume or --initialize-from when training')
    if not args.resume and (output/'last.pt').exists(): raise FileExistsError('Existing run: use a new output or --resume')
    plan=batch_plan(world,cfg.get('global_batch',6),cfg.get('micro_batch',1),world)
    if world>1 and use_cuda and cfg.get('sync_batchnorm',True): model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    raw=model
    if world>1: model=DDP(model,device_ids=[local] if use_cuda else None,broadcast_buffers=False)
    base,translation,mapping=[],[],[]
    for name,param in raw.named_parameters():
        if not param.requires_grad: continue
        (translation if name.startswith('translation.') else mapping if name.startswith('mapping.') else base).append(param)
    groups=[{'params':base,'lr':cfg.get('lr',1e-4)}]
    if translation: groups.append({'params':translation,'lr':cfg.get('translation_lr',1e-5)})
    if mapping: groups.append({'params':mapping,'lr':cfg.get('mapping_lr',1e-3)})
    optimizer=torch.optim.Adam(groups,weight_decay=cfg.get('weight_decay',1e-4))
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,cfg.get('milestones',[20,35]),gamma=cfg.get('gamma',.2))
    start,best=0,None
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start=checkpoint['epoch']+int(checkpoint.get('epoch_complete',True))
        best=checkpoint['best']
        # A best commit may succeed immediately before a last commit fails.
        # Preserve that measured dev optimum when resuming the older last state.
        if (output/'best.pt').is_file():
            saved_best=torch.load(output/'best.pt',map_location='cpu',weights_only=False)
            if saved_best.get('manifests')!=manifests or saved_best.get('vocab')!=vocab:
                raise ValueError('Best checkpoint belongs to different data/vocab')
            score=saved_best.get('best')
            if score is not None:
                selection=cfg.get('selection_metric','wer')
                best=score if best is None else (min(best,score) if selection=='wer' else max(best,score))
            del saved_best
        for key in ('global_batch','seed','code_sha256'):
            if cfg.get(key)!=checkpoint['config'].get(key): raise ValueError(f'Resume config changed: {key}')
        if world==checkpoint['world_size']: restore_rng(checkpoint['rng'][rank])
        else:
            seed_everything(seed+start*1000003+rank)
            if rank==0: print('GPU count changed: resume is NOT bitwise equivalent.',flush=True)
    else: seed_everything(seed+rank)
    if rank==0:
        write_json(output/'run_metadata.json',{'config':cfg,'batch_plan':plan,'environment':environment(),
                   'manifests':manifests,'resumed_from':args.resume,'initialized_from':args.initialize_from,
                   'status':'reconstruction; NOT verified paper reproduction','upstream_commit':'af5e8475d755b9d2e92c0142c8b7084651c3a4ee'})
        write_json(output/'vocab.json',vocab)
        print(json.dumps(plan),flush=True)
    dataset=SignDataset(cfg['train'],cfg['data_root'],train=True,seed=seed,input_kind=cfg.get('input_kind','rgb'),target_field=cfg.get('target_field','gloss'))
    accumulation=plan['accumulation']
    store=make_store(cfg.get('persistence')) if rank==0 else None
    session_begin=time.monotonic()
    session_updates=0
    interrupted=[False]
    def request_stop(*_): interrupted[0]=True
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    def save_progress(epoch,updates,complete,improved=False):
        states=gather_object(rng_state(),world)
        error=None
        if rank==0:
            try:
                state={'model':raw.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                       'scaler':scaler.state_dict(),'rng':states,'epoch':epoch,'epoch_complete':complete,
                       'optimizer_steps_in_epoch':updates,'best':best,'config':cfg,
                       'vocab':vocab,'world_size':world,'manifests':manifests}
                atomic_save(state,output/'last.pt')
                if improved:
                    atomic_save(state,output/'best.pt')
                    if store: publish(store,output/'best.pt','best')
                # Last is committed only after its new best is available remotely.
                if store: publish(store,output/'last.pt')
            except Exception as exc: error=f'Checkpoint persistence failed: {type(exc).__name__}: {exc}'
        errors=gather_object(error,world)
        if any(errors): raise RuntimeError(next(e for e in errors if e))
    for epoch in range(start,cfg.get('epochs',50)):
        begin=time.time()
        dataset.epoch=epoch
        updates=checkpoint.get('optimizer_steps_in_epoch',0) if args.resume and epoch==start and not checkpoint.get('epoch_complete',True) else 0
        sampler=GlobalBatchSampler(len(dataset),plan['global_batch'],rank,world,seed,epoch,updates)
        loader=DataLoader(dataset,batch_size=plan['micro_batch'],sampler=sampler,collate_fn=collate,
                          num_workers=cfg.get('workers',2),pin_memory=use_cuda,
                          generator=torch.Generator().manual_seed(seed+epoch))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total=0.
        if store is not None or (world>1 and cfg.get('persistence')):
            save_progress(epoch,updates,False)
        for step,batch in enumerate(loader):
            sync=(step+1)%accumulation==0
            with model.no_sync() if world>1 and not sync else nullcontext():
                with amp(): loss=model(batch['video'].to(device,non_blocking=True),batch['lengths'],batch['rows'])
                finite=torch.isfinite(loss.detach()).to(torch.int32)
                if world>1: dist.all_reduce(finite,op=dist.ReduceOp.MIN)
                if not finite.item(): raise FloatingPointError('Nonfinite loss: stopping all ranks')
                scaler.scale(loss/accumulation).backward()
            total+=loss.detach().item()
            if sync:
                if cfg.get('grad_clip',0):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                updates+=1
                session_updates+=1
                minutes=cfg.get('max_session_minutes',0)
                steps_limit=cfg.get('max_session_steps',0)
                stop=torch.tensor(int(interrupted[0] or (minutes and time.monotonic()-session_begin>=minutes*60)
                                      or (steps_limit and session_updates>=steps_limit)),device=device)
                if world>1: dist.all_reduce(stop,op=dist.ReduceOp.MAX)
                interval=cfg.get('checkpoint_every_updates',0)
                if stop.item() or (interval and updates%interval==0): save_progress(epoch,updates,False)
                if stop.item():
                    if rank==0: print('SESSION_SAVED: run the notebook again to continue.',flush=True)
                    if world>1: dist.destroy_process_group()
                    return
            if rank==0 and step%20==0: print(f'epoch={epoch+1} micro_step={step+1}/{len(loader)} loss={loss.item():.5f}',flush=True)
        scheduler.step()
        # Common buffers before independent, unevenly-sharded evaluation.
        if world>1:
            for buffer in raw.buffers(): dist.broadcast(buffer,0)
        scores,predictions=evaluate(raw,cfg,vocab,'dev',device,rank,world,amp)
        metric=cfg.get('selection_metric','wer')
        score=scores[cfg.get('eval_head','sequence')]['wer'] if metric=='wer' else scores[metric]
        improved=best is None or (score<best if metric=='wer' else score>best)
        if improved: best=score
        save_progress(epoch,updates,True,improved)
        mean=torch.tensor([total,len(loader)],device=device,dtype=torch.float64)
        if world>1: dist.all_reduce(mean)
        if rank==0:
            if improved:
                write_json(output/'best_dev_predictions.json',predictions)
                write_json(output/'best_dev_metrics.json',scores)
            record={'epoch':epoch+1,'loss':(mean[0]/mean[1]).item() if mean[1] else None,'dev':scores,'best':best,
                    'seconds':time.time()-begin,'dropped_train_samples':len(dataset)%plan['global_batch']}
            with (output/'history.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
        if world>1: dist.barrier()
    if world>1: dist.destroy_process_group()


if __name__=='__main__': main()
