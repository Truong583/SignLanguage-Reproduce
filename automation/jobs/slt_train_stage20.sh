#!/usr/bin/env bash
set -euo pipefail
ROOT=${SLT_WORKSPACE:?Run through installed worker}
JOB_ID=${SLT_JOB_ID:?Missing job ID}
LABEL=${SLT_WORKER_LABEL:?Missing worker label}
SOURCE="$ROOT/.updates/releases/9c2792dafbb866649a9036433ed851b1e113b1a3"
CHECKPOINT="$ROOT/runs/phoenix14t_cslr_suite_9c2792dafbb8/phoenix14t_cslr"
OUTPUT="$ROOT/runs/phoenix14t_slt_cc25_cpuadam_v1"
IMAGE=mixsigngraph-repro:7d49da5a013e9554
GPU_NAME="slttrain-${JOB_ID}"
WATCH_NAME="sltwatch-${JOB_ID}"
cd "$ROOT"
docker info --format '{{.OSType}}'
# The prior bounded job must have actually passed on this host.
python3 - <<'GATE_PY'
import json
from pathlib import Path
values=list(Path('runs').glob('slt-audit-20261019-v3-*/report/result.json'))
if not any(json.loads(p.read_text()).get('status')=='passed_slt_runtime_sanity_only' for p in values):
    raise SystemExit('Previous SLT runtime audit did not pass; no training launched')
GATE_PY
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
test -z "$BUSY" || { echo 'GPU busy; refusing to overlap another training process.'; exit 2; }
JOB=$(mktemp -d "$ROOT/runs/${JOB_ID}-XXXXXX")
mkdir -p "$JOB/code" "$JOB/report" "$JOB/spool" "$OUTPUT/metrics"
chmod 700 "$JOB" "$JOB/code" "$JOB/report" "$JOB/spool" "$OUTPUT" "$OUTPUT/metrics"
cat > "$JOB/code/run_slt_long.py" <<'SLT_TRAIN_PY'
"""Initialize once, prove full checkpoint resume, then train the requested SLT run."""
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import torch
import yaml
from slt_checkpoint import write_json

active=None
stopping=False
def stop(*unused):
    global stopping
    stopping=True
    if active is not None and active.poll() is None: active.terminate()
signal.signal(signal.SIGTERM,stop)
signal.signal(signal.SIGINT,stop)

output = Path('/output')
output.mkdir(exist_ok=True)
lock = (output / 'training.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',
    '/job/test_cpu_state_adam.py','/job/test_slt_recipe_audit.py','/job/test_slt_long_resume.py'],check=True)
source = Path('/checkpoint/best.pt')
base = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
if base['epoch'] + 1 != 22:
    raise ValueError('Expected the finalized CSLR epoch 22')
cfg = dict(base['config'])
for split in ('train', 'dev', 'test'):
    if hashlib.sha256(Path(cfg[split]).read_bytes()).hexdigest() != base['manifests'][split]:
        raise ValueError('Dataset manifest differs from finalized CSLR')
del base
gc.collect()
recipe_files = ['slt_train_engine.py','cpu_state_adam.py','audited_slt_model.py','slt_checkpoint.py']
fingerprint = hashlib.sha256(b''.join((Path('/job') / name).read_bytes() for name in recipe_files)).hexdigest()
cfg.update(task='slt', output=str(output), mbart_path='/workspace/assets/mbart-large-cc25',
    language='de_DE', translation_checkpointing=True, translation_weight=1., label_smoothing=.2,
    translation_lr=1e-5, mapping_lr=1e-3, lr=1e-4, weight_decay=1e-4,
    conv_ctc=1., seq_ctc=1., distillation=25., milestones=[20,35], gamma=.2,
    text_beam=6, text_max_length=60, text_length_penalty=1., text_level='word', selection_metric='rouge',
    epochs=20, seed=0, global_batch=6, micro_batch=1, eval_batch=1, workers=0, cpu_threads=2,
    precision='fp16', activation_offload='cpu_checkpoint', low_memory_loader=True,
    checkpoint_every_updates=100, max_session_minutes=6*24*60, max_session_steps=0,
    code_sha256=fingerprint, persistence=None)
if shutil.disk_usage(output).free < 30*1024**3:
    raise OSError('Need at least 30 GiB free before SLT starts, including checkpoint reserve')
config = output / 'local_config.yaml'
last = output / 'last.pt'

def invoke(arguments, limit=0):
    global active
    if stopping: raise SystemExit(75)
    cfg['max_session_steps'] = limit
    config.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    active=subprocess.Popen([sys.executable, '-u', '/job/slt_train_engine.py', '--config', str(config), *arguments])
    code=active.wait()
    active=None
    if stopping: raise SystemExit(75)
    if code: raise subprocess.CalledProcessError(code,'slt_train_engine')

print('SLT: run through epoch 20, then pause; original 50-epoch schedule retained for later continuation.', flush=True)
print('CSLR checkpoint stays read-only. Adam moments stay on CPU. GPU allocator limit: 85%.', flush=True)
if not last.exists():
    print('TRAIN_START: first real optimizer update followed by durable checkpoint and restart.', flush=True)
    invoke(['--initialize-from', str(source)], limit=1)

saved = torch.load(last, map_location='cpu', weights_only=False, mmap=True)
if saved['config'].get('code_sha256') != fingerprint:
    raise ValueError('Training recipe changed: explicit checkpoint compatibility review required')
for name in ('task','seed','global_batch','micro_batch','precision','translation_lr','mapping_lr','lr',
             'weight_decay','conv_ctc','seq_ctc','distillation','milestones','gamma','label_smoothing',
             'translation_weight','language','mbart_path','text_beam','text_max_length','selection_metric'):
    if saved['config'].get(name) != cfg.get(name):
        raise ValueError('Resume recipe changed: '+name)
if saved.get('world_size') != 1 or not saved['optimizer']['state']:
    raise ValueError('Incomplete SLT optimizer checkpoint')
complete = saved['epoch_complete'] and saved['epoch'] + 1 >= cfg['epochs']
print(f'CHECKPOINT_READBACK_PASS: epoch={saved["epoch"]+1}, updates={saved["optimizer_steps_in_epoch"]}', flush=True)
del saved
gc.collect()
if not complete:
    invoke(['--resume', str(last)])
saved = torch.load(last, map_location='cpu', weights_only=False, mmap=True)
complete = saved['epoch_complete'] and saved['epoch'] + 1 >= cfg['epochs']
del saved
gc.collect()
if not complete:
    print('SESSION_SAVED: partial SLT checkpoint retained; next approved job can resume.', flush=True)
    raise SystemExit(75)
stage={'epochs_completed':20,'full_schedule_epochs':50,'selection':'dev_rouge',
    'test_evaluated':False,'resume_available':True,'exact_paper_reproduction':False}
write_json(output/'stage20_completed.json',stage)
write_json(output/'metrics'/'stage20_completed.json',stage)
print('SLT_STAGE20_COMPLETE: stopped after epoch 20 and dev evaluation; last/best retained; no test evaluation.', flush=True)
SLT_TRAIN_PY
cat > "$JOB/code/slt_train_engine.py" <<'SLT_TRAIN_PY'
"""Single-node DDP, exact effective-batch arithmetic, epoch-boundary resume."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time
import signal
import gc

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler, Subset
import yaml

from repro.data import SignDataset, collate, read_manifest, build_vocab
from audited_slt_model import AuditedSLTModel as SignModel
from cpu_state_adam import CPUStateAdam
from repro.metrics import ctc_decode, corpus_wer, translation_scores
from repro.runtime import batch_plan, seed_everything, rng_state, restore_rng, atomic_save, sha256, environment, write_json, loader_options
from repro.persistence import make_store, publish


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
    loader = DataLoader(subset,batch_size=cfg.get('eval_batch',1),
                        collate_fn=collate,**loader_options(cfg,device.type=='cuda'))
    model.eval()
    predictions=[]
    with torch.no_grad():
        for index,batch in enumerate(loader):
            with amp():
                out = model.features(batch['video'].to(device,non_blocking=True),batch['lengths'])
                texts = model.generate(out)
            if index%20==0:
                print(f'EVAL {split}: {index+1}/{len(loader)}',flush=True)
                write_json(Path(cfg['output'])/'progress.json',{'phase':split,'samples_done':index+1,'samples_total':len(loader),'time':time.time()})
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


from slt_checkpoint import atomic_save, write_json, mirror_history


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
    if use_cuda:
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(.85, device)
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
    scaler=torch.amp.GradScaler('cuda',enabled=precision=='fp16', init_scale=1.)
    output=Path(cfg['output'])
    if rank==0: output.mkdir(parents=True,exist_ok=True)
    if world>1: dist.barrier()
    checkpoint_path=args.resume or args.checkpoint or args.initialize_from
    # Only own/trusted checkpoints. Optimizer and RNG require weights_only=False.
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False,mmap=True) if checkpoint_path else None
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
    optimizer=CPUStateAdam(groups,weight_decay=cfg.get('weight_decay',1e-4))
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
            saved_best=torch.load(output/'best.pt',map_location='cpu',weights_only=False,mmap=True)
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
        print('Activation storage: '+(raw.activation_offload if use_cuda else 'none (CPU execution)'),flush=True)
        print('Data loader: '+json.dumps(loader_options(cfg,use_cuda)),flush=True)
    resume_cursor=checkpoint.get('optimizer_steps_in_epoch',0) if args.resume and not checkpoint.get('epoch_complete',True) else 0
    resume_sum=checkpoint.get('epoch_loss_sum',0.) if resume_cursor else 0.
    resume_count=checkpoint.get('epoch_loss_count',0) if resume_cursor else 0
    checkpoint=None
    gc.collect()
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
                       'epoch_loss_sum':total,'epoch_loss_count':count,
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
        updates=resume_cursor if epoch==start else 0
        initial_updates=updates
        sampler=GlobalBatchSampler(len(dataset),plan['global_batch'],rank,world,seed,epoch,updates)
        loader=DataLoader(dataset,batch_size=plan['micro_batch'],sampler=sampler,collate_fn=collate,
                          **loader_options(cfg,use_cuda),
                          generator=torch.Generator().manual_seed(seed+epoch))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total=resume_sum if epoch==start else 0.
        count=resume_count if epoch==start else 0
        if store is not None or (world>1 and cfg.get('persistence')):
            save_progress(epoch,updates,False)
        for step,batch in enumerate(loader):
            if rank==0 and (step%20==0 or int(batch['lengths'].max())>=512):
                print('TRAIN_SAMPLE: '+json.dumps({'epoch':epoch+1,'micro_step':initial_updates*accumulation+step+1,
                      'ids':[row['id'] for row in batch['rows']],
                      'video_shape':list(batch['video'].shape)}),flush=True)
            sync=(step+1)%accumulation==0
            with model.no_sync() if world>1 and not sync else nullcontext():
                try:
                    with amp(): loss=model(batch['video'].to(device,non_blocking=True),batch['lengths'],batch['rows'])
                except torch.OutOfMemoryError:
                    print('CUDA_OOM sample details: '+json.dumps({'ids':[row['id'] for row in batch['rows']],
                          'video_shape':list(batch['video'].shape),'padded_lengths':batch['lengths'].tolist(),
                          'activation_storage':raw.activation_offload}),flush=True)
                    raise
                finite=torch.isfinite(loss.detach()).to(torch.int32)
                if world>1: dist.all_reduce(finite,op=dist.ReduceOp.MIN)
                if not finite.item(): raise FloatingPointError('Nonfinite loss: stopping all ranks')
                scaler.scale(loss/accumulation).backward()
            total+=loss.detach().item()
            count+=1
            if sync:
                scaler.unscale_(optimizer)
                if any(p.grad is not None and not torch.isfinite(p.grad).all().item() for p in model.parameters()):
                    raise FloatingPointError('Nonfinite gradient; retained last checkpoint, stopping for diagnosis')
                if cfg.get('grad_clip',0):
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
                if stop.item() or updates==1 or (interval and updates%interval==0): save_progress(epoch,updates,False)
                if stop.item():
                    if rank==0: print('SESSION_SAVED: run the notebook again to continue.',flush=True)
                    if world>1: dist.destroy_process_group()
                    return
            if rank==0 and step%20==0:
                micro=initial_updates*accumulation+step+1
                maximum=len(dataset)//plan['global_batch']*plan['global_batch']//plan['micro_batch']
                print(f'epoch={epoch+1} micro_step={micro}/{maximum} loss={loss.item():.5f}',flush=True)
                write_json(output/'progress.json',{'phase':'train','epoch':epoch+1,'epochs':cfg.get('epochs',50),
                    'micro_step':micro,'micro_steps':maximum,'loss':loss.item(),'time':time.time()})
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
        mean=torch.tensor([total,count],device=device,dtype=torch.float64)
        if world>1: dist.all_reduce(mean)
        if rank==0:
            if improved:
                write_json(output/'best_dev_predictions.json',predictions)
                write_json(output/'best_dev_metrics.json',scores)
            record={'epoch':epoch+1,'loss':(mean[0]/mean[1]).item() if mean[1] else None,'dev':scores,'best':best,
                    'seconds':time.time()-begin,'dropped_train_samples':len(dataset)%plan['global_batch']}
            with (output/'history.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(record)+'\n')
            mirror_history(output)
            print(json.dumps(record),flush=True)
        if world>1: dist.barrier()
    if world>1: dist.destroy_process_group()


if __name__=='__main__': main()
SLT_TRAIN_PY
cat > "$JOB/code/slt_checkpoint.py" <<'SLT_TRAIN_PY'
"""Bounded, durable checkpoint writes for the isolated SLT run."""
import os
import json
from pathlib import Path
import shutil
import torch


def write_json(path, value):
    path=Path(path)
    text=json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    targets=[path]
    if path.name in ('progress.json','best_dev_metrics.json','test_metrics.json','completed.json'):
        targets.append(path.parent/'metrics'/path.name)
    for target in targets:
        target.parent.mkdir(exist_ok=True)
        temporary=target.with_suffix(target.suffix+'.tmp')
        temporary.write_text(text,encoding='utf-8')
        os.replace(temporary,target)


def mirror_history(output):
    output=Path(output)
    folder=output/'metrics'
    folder.mkdir(exist_ok=True)
    temporary=folder/'history.jsonl.tmp'
    shutil.copyfile(output/'history.jsonl',temporary)
    os.replace(temporary,folder/'history.jsonl')


def atomic_save(state, path):
    path = Path(path)
    # Keep room for a complete temporary checkpoint and the host's other work.
    tensors = {}
    def visit(value):
        if isinstance(value, torch.Tensor):
            tensors[(value.device, value.untyped_storage().data_ptr())] = value.untyped_storage().nbytes()
        elif isinstance(value, dict):
            for item in value.values(): visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value: visit(item)
    visit(state)
    required = sum(tensors.values()) + 1024**3
    if shutil.disk_usage(path.parent).free < required + 10 * 1024**3:
        raise OSError('Not enough free disk for checkpoint plus 10 GiB reserve; prior checkpoints retained')
    temporary = path.with_suffix('.pt.tmp')
    with temporary.open('wb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    # Metadata and tensor shape round trip without a second full RAM copy.
    loaded = torch.load(temporary, map_location='cpu', weights_only=False, mmap=True)
    assert loaded['epoch'] == state['epoch']
    assert loaded['optimizer_steps_in_epoch'] == state['optimizer_steps_in_epoch']
    assert set(loaded['model']) == set(state['model'])
    assert all(loaded['model'][k].shape == v.shape for k, v in state['model'].items())
    del loaded
    os.replace(temporary, path)
    if os.name == 'posix':
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
    print(f'CHECKPOINT_SAVED: {path.name}; epoch={state["epoch"]+1}; '
          f'updates={state["optimizer_steps_in_epoch"]}; complete={state["epoch_complete"]}', flush=True)
SLT_TRAIN_PY
cat > "$JOB/code/cpu_state_adam.py" <<'SLT_TRAIN_PY'
"""Diagnostic Adam offload: CPU moments and bounded CPU update chunks.

Uses public torch.optim.Adam for the update itself (coupled weight decay,
not AdamW). Model parameters and gradients remain on their original device.
CPU/GPU arithmetic need not be bitwise identical. Not yet validated on the
teacher's GPU or integrated into the production training/resume runner.
"""
import copy
import torch


class CPUStateAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, chunk_elements=262144):
        if not isinstance(chunk_elements, int) or chunk_elements <= 0:
            raise ValueError('chunk_elements must be a positive integer')
        self.chunk_elements = chunk_elements
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        for group in self.param_groups:
            self._validate_group(group)
            for param in group['params']:
                self._validate_param(param)

    @staticmethod
    def _validate_group(group):
        if group['lr'] < 0 or group['eps'] < 0 or group['weight_decay'] < 0:
            raise ValueError('Invalid Adam hyperparameters')
        if not all(0 <= beta < 1 for beta in group['betas']):
            raise ValueError('Invalid Adam betas')
        for flag in ('amsgrad', 'maximize', 'differentiable', 'capturable', 'fused'):
            if group.get(flag, False):
                raise ValueError('Unsupported Adam option: ' + flag)

    @staticmethod
    def _validate_param(param):
        if param.dtype not in (torch.float32, torch.float64) or not param.is_contiguous():
            raise ValueError('CPUStateAdam requires contiguous FP32/FP64 parameters')

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            self._validate_group(group)
            for param in group['params']:
                if param.grad is None:
                    continue
                self._validate_param(param)
                if param.grad.is_sparse or not param.grad.is_contiguous():
                    raise ValueError('CPUStateAdam requires dense contiguous gradients')
                state = self.state[param]
                if not state:
                    state['step'] = torch.tensor(0., device='cpu')
                    state['exp_avg'] = torch.zeros_like(param, device='cpu')
                    state['exp_avg_sq'] = torch.zeros_like(param, device='cpu')
                flat = param.view(-1)
                grads = param.grad.view(-1)
                mean = state['exp_avg'].view(-1)
                variance = state['exp_avg_sq'].view(-1)
                for start in range(0, param.numel(), self.chunk_elements):
                    end = min(start + self.chunk_elements, param.numel())
                    # copy=True is required even in CPU tests: weight decay must
                    # not mutate the original gradient, and writes occur below.
                    chunk = flat[start:end].detach().to('cpu', copy=True)
                    chunk.grad = grads[start:end].detach().to('cpu', copy=True)
                    update = torch.optim.Adam([chunk], lr=group['lr'], betas=group['betas'],
                                              eps=group['eps'], weight_decay=group['weight_decay'],
                                              foreach=False)
                    update.state[chunk] = {'step': state['step'].clone(),
                                           'exp_avg': mean[start:end],
                                           'exp_avg_sq': variance[start:end]}
                    update.step()
                    flat[start:end].copy_(chunk)
                    del update, chunk
                # Every chunk uses the same previous counter; increment once.
                state['step'].add_(1)
        return loss

    def load_state_dict(self, saved):
        # The standard Optimizer loader would move moments onto the GPU.
        if len(saved['param_groups']) != len(self.param_groups):
            raise ValueError('Optimizer parameter group count differs')
        groups, states = [], {}
        for source, target in zip(saved['param_groups'], self.param_groups):
            if len(source['params']) != len(target['params']):
                raise ValueError('Optimizer parameter count differs')
            group = copy.deepcopy({k: v for k, v in source.items() if k != 'params'})
            group['params'] = target['params']
            self._validate_group(group)
            groups.append(group)
            for ident, param in zip(source['params'], target['params']):
                value = saved['state'].get(ident)
                if not value:
                    continue
                if set(value) != {'step', 'exp_avg', 'exp_avg_sq'}:
                    raise ValueError('Unexpected Adam checkpoint state')
                if any(value[k].shape != param.shape for k in ('exp_avg', 'exp_avg_sq')):
                    raise ValueError('Optimizer state tensor shape differs')
                step = torch.as_tensor(value['step']).detach().to('cpu')
                if step.numel() != 1 or not torch.isfinite(step).all() or step.item() < 0 or step.item() != int(step.item()):
                    raise ValueError('Invalid Adam step counter')
                states[param] = {'step': step.clone(),
                                 'exp_avg': value['exp_avg'].detach().to(device='cpu', dtype=param.dtype).clone(),
                                 'exp_avg_sq': value['exp_avg_sq'].detach().to(device='cpu', dtype=param.dtype).clone()}
        self.state.clear()
        self.state.update(states)
        self.param_groups[:] = groups
SLT_TRAIN_PY
cat > "$JOB/code/audited_slt_model.py" <<'SLT_TRAIN_PY'
"""Candidate SLT recipe aligned to available upstream translation source.

Still uses full CC25 vocabulary: author-pruned model/map_ids.pkl are missing.
This is NOT an assertion of exact paper reproduction. CSLR weights untouched.
"""
import math
import torch
from torch.nn import functional as F
from repro.model import SignModel
from repro.data import encode_targets


def translation_targets(ids, pad_id, lang_id):
    """Start decoder with language ID; exclude language and padding from loss."""
    if ids.ndim != 2 or ids.shape[1] < 2:
        raise ValueError('Expected mBART token sequences')
    labels = ids.clone()
    labels[(labels == pad_id) | (labels == lang_id)] = -100
    decoder = torch.full_like(ids, pad_id)
    decoder[:, 0] = lang_id
    shifted = ids[:, :-1].clone()
    shifted[shifted == lang_id] = pad_id
    decoder[:, 1:] = shifted
    return labels, decoder


def upstream_smoothed_kl(logits, labels, pad_id, smoothing):
    """Upstream XentLoss distribution, without allocating a dense target matrix.

Gold has mass 1-eps; pad has zero; remaining V-2 entries have eps/(V-2).
Preserves the KL entropy constant and the sum reduction of upstream XentLoss.
"""
    if not 0 <= smoothing < 1:
        raise ValueError('Invalid smoothing')
    size = logits.shape[-1]
    if size < 3 or not 0 <= pad_id < size:
        raise ValueError('Invalid vocabulary')
    logp = F.log_softmax(logits.float(), dim=-1)
    valid = (labels != -100) & (labels != pad_id)
    safe = labels.masked_fill(~valid, 0)
    nll = -logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    if smoothing == 0:
        return nll.masked_select(valid).sum()
    confidence = 1 - smoothing
    other = smoothing / (size - 2)
    smooth = -logp.sum(-1) + logp[..., pad_id]
    entropy_term = confidence * math.log(confidence) + smoothing * math.log(other)
    values = (confidence - other) * nll + other * smooth + entropy_term
    return values.masked_select(valid).sum()


class AuditedSLTModel(SignModel):
    def forward(self, video, lengths, rows=None):
        out = self.features(video, lengths)
        if rows is None:
            return out
        labels, label_lengths, required = encode_targets(rows, self.vocab, self.cfg.get('target_field', 'gloss'))
        if (required > out['lengths']).any():
            raise ValueError('CTC alignment impossible; labels are not truncated')
        loss = out['sequence'].float().sum() * 0
        for head, weight in [('conv', self.cfg.get('conv_ctc', 1.)), ('sequence', self.cfg.get('seq_ctc', 1.))]:
            if weight:
                loss = loss + weight * F.ctc_loss(out[head].float().log_softmax(-1), labels,
                    out['lengths'], label_lengths, blank=0, reduction='none', zero_infinity=False).mean()
        if self.cfg.get('distillation', 25.):
            loss = loss + self.cfg.get('distillation', 25.) * self.kd(
                out['conv'].float(), out['sequence'].detach().float(), use_blank=False)
        if self.translation is not None:
            ids = self.tokenizer([row['text'] for row in rows], return_tensors='pt', padding=True).input_ids.to(video.device)
            labels, decoder = translation_targets(ids, self.tokenizer.pad_token_id,
                self.tokenizer.lang_code_to_id[self.cfg.get('language', 'de_DE')])
            translated = self.translation(**self.translation_inputs(out['features'], out['lengths']),
                                          decoder_input_ids=decoder)
            loss = loss + self.cfg.get('translation_weight', 1.) * upstream_smoothed_kl(
                translated.logits, labels, self.tokenizer.pad_token_id, self.cfg.get('label_smoothing', .2)) / len(rows)
        return loss

    @torch.no_grad()
    def generate(self, out):
        if self.translation is None:
            return [''] * len(out['lengths'])
        generated = self.translation.generate(**self.translation_inputs(out['features'], out['lengths']),
            decoder_start_token_id=self.tokenizer.lang_code_to_id[self.cfg.get('language', 'de_DE')],
            num_beams=self.cfg.get('text_beam', 6), max_length=self.cfg.get('text_max_length', 60),
            length_penalty=self.cfg.get('text_length_penalty', 1.), use_cache=True)
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)
SLT_TRAIN_PY
cat > "$JOB/code/slt_long_supervisor.py" <<'SLT_TRAIN_PY'
"""Keep full logs on the host; stop cleanly and expose final outcome."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

root=Path('/report')
started=time.time()
child=None
stopping=False
def stop(*unused):
    global stopping
    stopping=True
    if child is not None and child.poll() is None: child.terminate()
signal.signal(signal.SIGTERM,stop)
signal.signal(signal.SIGINT,stop)
code=1
try:
    with (root/'train.log').open('a',encoding='utf-8',buffering=1) as log:
        child=subprocess.Popen([sys.executable,'-u','/job/run_slt_long.py'],stdout=log,stderr=subprocess.STDOUT,
                               start_new_session=True)
        def stop_group(*unused):
            global stopping
            stopping=True
            if child.poll() is None: os.killpg(child.pid,signal.SIGTERM)
        signal.signal(signal.SIGTERM,stop_group)
        signal.signal(signal.SIGINT,stop_group)
        code=child.wait()
        log.write('\nSLT_EXIT: '+str(code)+'\n')
finally:
    # A normal end must include completion metadata, not merely a zero return.
    completed=Path('/output/stage20_completed.json').is_file() and code==0
    result={'status':'stage20_completed' if completed else 'interrupted' if stopping or code==75 else 'failed',
            'exit_code':code,'seconds':round(time.time()-started,1),
            'checkpoint_exists':Path('/output/last.pt').is_file(),
            'exact_paper_reproduction':False}
    temporary=root/'result.json.tmp'
    temporary.write_text(json.dumps(result,indent=2),encoding='utf-8')
    os.replace(temporary,root/'result.json')
sys.exit(code)
SLT_TRAIN_PY
cat > "$JOB/code/slt_long_observer.py" <<'SLT_TRAIN_PY'
"""Separate CPU/network observer: never receives dataset or model checkpoints."""
import json
import os
from pathlib import Path
import time
from scripts.diagnostics import redact
from scripts.privacy import require_private

report,spool,metrics=Path('/report'),Path('/spool'),Path('/metrics')
cfg=json.loads(Path('/machine.json').read_text())
key=Path('/wandb_key').read_text().strip()
os.environ.update(WANDB_API_KEY=key,WANDB_CONSOLE='off',WANDB_DISABLE_CODE='true',
    WANDB_DISABLE_GIT='true',WANDB_ERROR_REPORTING='false',WANDB_DIR=str(spool),
    WANDB_CACHE_DIR=str(spool/'cache'),WANDB_DATA_DIR=str(spool/'staging'),WANDB_CONFIG_DIR='/tmp/wandb')
import wandb
run=None
seen=set()
last_progress=None
finished_at=None
deadline=time.monotonic()+7*24*3600

def upload(name,content):
    target=spool/name
    target.write_text(redact(content),encoding='utf-8')
    run.save(str(target),base_path=str(spool),policy='now')

while time.monotonic()<deadline:
    try:
        require_private(cfg['wandb_entity'],cfg['wandb_project'],key)
        if run is None:
            run=wandb.init(entity=cfg['wandb_entity'],project=cfg['wandb_project'],
                id=os.environ['SLT_RUN_ID'],name=os.environ['SLT_RUN_ID'],resume='allow',
                job_type='slt-training',dir=str(spool),settings=wandb.Settings(init_timeout=60))
            run.config.update({'task':'SLT','dataset':'PHOENIX14T','epochs':20,'full_schedule_epochs':50,'global_batch':6,
                'micro_batch':1,'initialize_cslr_epoch':22,'optimizer':'Adam CPU moments',
                'mbart':'full CC25','exact_paper_reproduction':False},allow_val_change=True)
            print('Dashboard: '+run.url,flush=True)
        logfile=report/'train.log'
        if logfile.exists():
            with logfile.open('rb') as stream:
                stream.seek(max(0,logfile.stat().st_size-512*1024))
                upload('live_log_tail.txt',stream.read().decode('utf-8',errors='replace'))
        progress=metrics/'progress.json'
        if progress.exists():
            # Writer uses atomic replacement; tolerate a delayed initial empty file.
            text=progress.read_text()
            value=json.loads(text)
            run.summary['phase']=value.get('phase','unknown')
            if text!=last_progress:
                logged={'progress/'+k:v for k,v in value.items() if isinstance(v,(int,float))}
                run.log(logged)
                last_progress=text
            upload('progress.json',text)
        history=metrics/'history.jsonl'
        if history.exists():
            text=history.read_text()
            for line in text.splitlines():
                try: row=json.loads(line)
                except json.JSONDecodeError: continue
                epoch=row['epoch']
                if epoch in seen: continue
                values={'epoch':epoch,'train/loss':row['loss'],'epoch/seconds':row['seconds']}
                for name,value in row['dev'].items():
                    if isinstance(value,(int,float)): values['dev/'+name]=value
                    elif isinstance(value,dict) and 'wer' in value: values['dev/'+name+'/wer']=value['wer']
                run.log(values)
                seen.add(epoch)
            upload('history.jsonl',text)
        for name in ('best_dev_metrics.json','stage20_completed.json'):
            path=metrics/name
            if path.exists() and path.stat().st_size:
                value=json.loads(path.read_text())
                upload(name,json.dumps(value,ensure_ascii=False))
                run.summary[name.removesuffix('.json')]=value
        result=report/'result.json'
        if result.exists():
            value=json.loads(result.read_text())
            run.summary.update(value)
            upload('result.json',json.dumps(value))
            run.finish(exit_code=0 if value.get('status')=='stage20_completed' else 1)
            print('SLT RESULT UPLOADED',flush=True)
            break
        run.summary['status']='running_or_preparing'
        time.sleep(30)
    except Exception:
        print('Upload unavailable; logs retained on host; retry in 60 seconds.',flush=True)
        time.sleep(60)
SLT_TRAIN_PY
cat > "$JOB/code/test_cpu_state_adam.py" <<'SLT_TRAIN_PY'
"""Numerical checks; does not launch the research model or touch its files."""
import copy
import pytest
import torch
from cpu_state_adam import CPUStateAdam


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('chunk', [1, 7, 10000])
def test_matches_adam_and_round_trip(dtype, chunk):
    torch.manual_seed(391)
    left = [torch.nn.Parameter(torch.randn(3, 11, dtype=dtype)),
            torch.nn.Parameter(torch.randn(9, dtype=dtype))]
    right = [torch.nn.Parameter(p.detach().clone()) for p in left]
    make = lambda params: [{'params': [params[0]], 'lr': 1e-4},
                           {'params': [params[1]], 'lr': 1e-3, 'weight_decay': 0.03}]
    reference = torch.optim.Adam(make(left), weight_decay=1e-4, foreach=False)
    candidate = CPUStateAdam(make(right), weight_decay=1e-4, chunk_elements=chunk)
    # Different CPU vector widths can round differently at chunk boundaries.
    tolerance = dict(rtol=2e-6, atol=2e-8) if dtype == torch.float32 else dict(rtol=1e-12, atol=1e-14)
    for index in range(8):
        for a, b in zip(left, right):
            a.grad = torch.randn_like(a)
            b.grad = a.grad.clone()
        if index == 2:
            left[1].grad = right[1].grad = None
        original_grads = [p.grad.clone() if p.grad is not None else None for p in right]
        reference.step()
        candidate.step()
        for a, b, gradient in zip(left, right, original_grads):
            torch.testing.assert_close(a, b, **tolerance)
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(reference.state[a][key], candidate.state[b][key], **tolerance)
                assert candidate.state[b][key].device.type == 'cpu'
            if gradient is not None:
                torch.testing.assert_close(gradient, b.grad, rtol=0, atol=0)
        if index == 3:
            checkpoint = copy.deepcopy(candidate.state_dict())
            candidate = CPUStateAdam(make(right), chunk_elements=chunk)
            candidate.load_state_dict(checkpoint)


def test_rejects_unsupported_and_wrong_state_shape():
    with pytest.raises(ValueError):
        CPUStateAdam([torch.nn.Parameter(torch.ones(3, dtype=torch.float16))])
    with pytest.raises(ValueError):
        CPUStateAdam([{'params': [torch.nn.Parameter(torch.ones(3))], 'amsgrad': True}])
    p = torch.nn.Parameter(torch.ones(3))
    optimizer = CPUStateAdam([p])
    p.grad = torch.ones_like(p)
    optimizer.step()
    saved = copy.deepcopy(optimizer.state_dict())
    saved['state'][0]['exp_avg'] = torch.zeros(4)
    with pytest.raises(ValueError):
        optimizer.load_state_dict(saved)


def test_resume_exactly_matches_uninterrupted_offload():
    torch.manual_seed(72)
    p = torch.nn.Parameter(torch.randn(57))
    first = CPUStateAdam([p], chunk_elements=13)
    p.grad = torch.randn_like(p)
    first.step()
    q = torch.nn.Parameter(p.detach().clone())
    second = CPUStateAdam([q], chunk_elements=13)
    second.load_state_dict(copy.deepcopy(first.state_dict()))
    for _ in range(4):
        p.grad = torch.randn_like(p)
        q.grad = p.grad.clone()
        first.step()
        second.step()
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(first.state[p][key], second.state[q][key], rtol=0, atol=0)
SLT_TRAIN_PY
cat > "$JOB/code/test_slt_recipe_audit.py" <<'SLT_TRAIN_PY'
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from audited_slt_model import AuditedSLTModel, translation_targets, upstream_smoothed_kl


def test_language_shift_and_labels_match_upstream():
    ids = torch.tensor([[5, 6, 2, 4], [7, 2, 4, 1]])
    labels, decoder = translation_targets(ids, 1, 4)
    assert labels.tolist() == [[5, 6, 2, -100], [7, 2, -100, -100]]
    assert decoder.tolist() == [[4, 5, 6, 2], [4, 7, 2, 1]]
    assert ids.tolist() == [[5, 6, 2, 4], [7, 2, 4, 1]]


@pytest.mark.parametrize('smoothing', [0., .2])
def test_kl_matches_actual_upstream_class_value_and_gradient(smoothing):
    # Extract the exact class without importing upstream's missing assets.
    import repro.model
    source = Path(repro.model.__file__).resolve().parents[1] / 'MixSignGraph/modules/translation.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'XentLoss')
    scope = {'torch': torch, 'nn': torch.nn, 'Variable': torch.autograd.Variable}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), scope)
    torch.manual_seed(2)
    left = torch.randn(2, 4, 13, requires_grad=True)
    right = left.detach().clone().requires_grad_(True)
    labels = torch.tensor([[3, 5, 2, -100], [6, 2, -100, -100]])
    original_labels = labels.masked_fill(labels == -100, 1)
    expected = scope['XentLoss'](pad_index=1, smoothing=smoothing)(left.log_softmax(-1), original_labels)
    actual = upstream_smoothed_kl(right, labels, 1, smoothing)
    torch.testing.assert_close(expected, actual, rtol=2e-6, atol=2e-6)
    expected.backward()
    actual.backward()
    torch.testing.assert_close(left.grad, right.grad, rtol=2e-5, atol=2e-7)


def test_ce_routes_gradients_into_video_mapping_and_mbart(monkeypatch):
    torch.set_num_threads(2)
    from transformers import MBartConfig, MBartForConditionalGeneration
    from repro.data import collate
    class Tokenizer:
        pad_token_id = 1
        eos_token_id = 2
        lang_code_to_id = {'de_DE': 4}
        def __call__(self, texts, **kwargs):
            return SimpleNamespace(input_ids=torch.tensor([[5, 6, 2, 4] for text in texts]))
        def batch_decode(self, values, **kwargs):
            return ['decoded' for row in values]
    config = MBartConfig(vocab_size=16, d_model=16, encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32,
        max_position_embeddings=64, pad_token_id=1, bos_token_id=0, eos_token_id=2)
    monkeypatch.setattr('repro.model.load_translation', lambda *args: (Tokenizer(), MBartForConditionalGeneration(config)))
    cfg = dict(task='slt', input_kind='features', feature_dim=8, hidden_size=16,
               mbart_path='unused', translation_checkpointing=False, conv_ctc=0., seq_ctc=0., distillation=0.,
               text_beam=2, text_max_length=5)
    model = AuditedSLTModel(cfg, ['<blank>', '<unk>', 'a', 'b'])
    batch = collate([(torch.randn(12, 8), dict(id='test', gloss='a b', text='sentence'))])
    batch['video'].requires_grad_()
    loss = model(batch['video'], batch['lengths'], batch['rows'])
    loss.backward()
    for value in [batch['video'].grad, model.mapping[0].weight.grad,
                  model.translation.model.shared.weight.grad]:
        assert value is not None and torch.isfinite(value).all() and value.abs().sum() > 0
    # Classifier is not on the translation path: feeding decoded glosses would
    # break the CE gradient route into the continuous visual feature network.
    model.eval()
    with torch.no_grad():
        assert model.generate(model.features(batch['video'], batch['lengths'])) == ['decoded']
SLT_TRAIN_PY
cat > "$JOB/code/test_slt_long_resume.py" <<'SLT_TRAIN_PY'
"""Exercise the real isolated training loop across an interrupted update."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
import yaml
import slt_train_engine as engine


class SmallModel(torch.nn.Module):
    def __init__(self, cfg, vocab, initialize=True):
        super().__init__()
        self.layer = torch.nn.Linear(3, 2)
        self.dropout = torch.nn.Dropout(.25)
        self.translation = None
        self.activation_offload = 'none'
    def forward(self, video, lengths, rows):
        return self.dropout(self.layer(video)).square().mean()


def test_interrupted_training_restores_model_optimizer_rng_and_loss(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, 'SignModel', SmallModel)
    monkeypatch.setattr(engine, 'evaluate', lambda *args: ({'sequence': {'wer': 50.}}, []))
    monkeypatch.setattr(engine, 'environment', lambda: {'test': True})
    import slt_checkpoint
    from collections import namedtuple
    usage = namedtuple('usage', 'total used free')
    monkeypatch.setattr(slt_checkpoint.shutil, 'disk_usage', lambda _: usage(200*1024**3, 0, 200*1024**3))
    rows = []
    for i in range(12):
        np.save(tmp_path / f'{i}.npy', np.full((12+i, 3), i/12, dtype=np.float32))
        rows.append({'id':str(i),'features':f'{i}.npy','gloss':'a','text':'a'})
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text('\n'.join(json.dumps(row) for row in rows))
    cfg = dict(device='cpu', task='cslr', input_kind='features', feature_dim=3, data_root=str(tmp_path),
        train=str(manifest), dev=str(manifest), test=str(manifest), workers=0, seed=17,
        global_batch=6, micro_batch=1, epochs=2, code_sha256='fixture', checkpoint_every_updates=1)
    config = tmp_path / 'config.yaml'
    def run(folder, limit=0, resume=False):
        cfg.update(output=str(tmp_path/folder), max_session_steps=limit)
        config.write_text(yaml.safe_dump(cfg))
        args=['engine','--config',str(config)]
        if resume: args += ['--resume',str(tmp_path/folder/'last.pt')]
        monkeypatch.setattr(sys,'argv',args)
        engine.main()
    run('continuous')
    run('resumed',1)
    partial=torch.load(tmp_path/'resumed/last.pt',weights_only=False)
    assert partial['optimizer_steps_in_epoch']==1 and not partial['epoch_complete']
    assert partial['epoch_loss_count']==6
    del partial
    run('resumed',resume=True)
    left=torch.load(tmp_path/'continuous/last.pt',weights_only=False)
    right=torch.load(tmp_path/'resumed/last.pt',weights_only=False)
    for key in left['model']:
        torch.testing.assert_close(left['model'][key],right['model'][key],rtol=0,atol=0)
    for key,value in left['optimizer']['state'].items():
        for field in value:
            torch.testing.assert_close(value[field],right['optimizer']['state'][key][field],rtol=0,atol=0)
    assert left['scheduler']==right['scheduler']
    assert left['epoch_loss_sum']==right['epoch_loss_sum']
    assert left['epoch_loss_count']==right['epoch_loss_count']==12
    assert left['epoch_complete'] and right['epoch_complete']
    histories = [[json.loads(line) for line in (tmp_path/folder/'history.jsonl').read_text().splitlines()]
                 for folder in ('continuous','resumed')]
    assert [r['loss'] for r in histories[0]] == [r['loss'] for r in histories[1]]
SLT_TRAIN_PY

COMMON=(--init --user "$(id -u):$(id -g)" --read-only --cap-drop ALL
 --security-opt no-new-privileges --pids-limit 512
 --label "signlanguage.slt_worker=$LABEL" --label "signlanguage.slt_job=$JOB_ID"
 --log-opt max-size=5m --log-opt max-file=2
 --mount "type=bind,src=$JOB/code,dst=/job,readonly"
 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2)
cleanup() { docker stop --time 60 "$GPU_NAME" "$WATCH_NAME" >/dev/null 2>&1 || true; }
trap cleanup TERM INT EXIT
docker run -d --name "$WATCH_NAME" "${COMMON[@]}" \
 --network bridge --memory 1g --memory-swap 1g --cpus 1 \
 --tmpfs /tmp:rw,nosuid,nodev,size=256m \
 --mount "type=bind,src=$JOB/report,dst=/report,readonly" \
 --mount "type=bind,src=$JOB/spool,dst=/spool" \
 --mount "type=bind,src=$OUTPUT/metrics,dst=/metrics,readonly" \
 --mount "type=bind,src=$ROOT/.updates/machine.json,dst=/machine.json,readonly" \
 --mount "type=bind,src=$ROOT/.updates/wandb_key,dst=/wandb_key,readonly" \
 -e PYTHONPATH=/opt/signlanguage -e SLT_RUN_ID="$JOB_ID" \
 --workdir /spool "$IMAGE" python -u /job/slt_long_observer.py
if ! docker run -d --name "$GPU_NAME" "${COMMON[@]}" \
 --gpus device=0 --network none --memory 21g --memory-swap 21g --cpus 2 \
 --shm-size 1g --stop-timeout 60 --tmpfs /tmp:rw,nosuid,nodev,size=512m \
 --mount "type=bind,src=$SOURCE,dst=/workspace,readonly" \
 --mount "type=bind,src=$ROOT/assets,dst=/workspace/assets,readonly" \
 --mount "type=bind,src=$ROOT/data,dst=/workspace/data,readonly" \
 --mount "type=bind,src=$CHECKPOINT,dst=/checkpoint,readonly" \
 --mount "type=bind,src=$OUTPUT,dst=/output" \
 --mount "type=bind,src=$JOB/report,dst=/report" \
 -e PYTHONPATH=/job:/workspace -e HF_HOME=/tmp/hf \
 --workdir /workspace "$IMAGE" python -u /job/slt_long_supervisor.py; then
 printf '%s\n' '{"status":"container_start_failed","exit_code":1}' > "$JOB/report/result.json"
 docker wait "$WATCH_NAME" || true
 exit 1
fi
code=$(docker wait "$GPU_NAME")
if ! test -f "$JOB/report/result.json"; then
 oom=$(docker inspect --format '{{.State.OOMKilled}}' "$GPU_NAME")
 printf '{"status":"container_exited_without_result","exit_code":%s,"docker_oom_killed":%s}\n' "$code" "$oom" > "$JOB/report/result.json.tmp"
 mv "$JOB/report/result.json.tmp" "$JOB/report/result.json"
fi
docker wait "$WATCH_NAME"
echo "SLT job ended; exit=$code; logs=$JOB/report; checkpoints=$OUTPUT"
exit "$code"
