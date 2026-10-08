"""Local/Colab/Kaggle campaign with completed-run skipping and committed checkpoints."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from repro.suite import experiments,BASE

FILES=['suite-config.yaml','history.jsonl','run_metadata.json','vocab.json',
       'best_dev_metrics.json','best_dev_predictions.json','dev_metrics.json','test_metrics.json',
       'dev_predictions.json','test_predictions.json','comparison.json','training.log','COMPLETED.json']


def atomic_json(path, value):
    path=Path(path); temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8'); os.replace(temp,path)


def signature(cfg):
    volatile={'data_root','train','dev','test','output','resnet_weights','backbone_weights',
              'persistence','max_session_minutes','checkpoint_every_updates','workers','cpu_threads'}
    content={k:v for k,v in cfg.items() if k not in volatile}
    return hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()


class Remote:
    def __init__(self, args):
        self.args=args; self.service=None
        if args.drive_folder:
            from scripts.kaggle_drive import service_from_json,ensure_folder
            self.service=service_from_json(Path(os.environ['SIGN_DRIVE_CREDENTIALS']).read_text())
            runs=ensure_folder(self.service,args.drive_folder,'runs')
            self.root=ensure_folder(self.service,runs,args.output.name)
        else:
            self.root=args.persist_root
            if self.root: self.root.mkdir(parents=True,exist_ok=True)

    def folder(self, ident):
        if self.service:
            from scripts.kaggle_drive import ensure_folder
            return ensure_folder(self.service,self.root,ident)
        if self.root:
            path=self.root/ident; path.mkdir(parents=True,exist_ok=True); return path
        return None

    def get(self, folder, name, destination):
        if not folder: return
        if self.service:
            from scripts.kaggle_drive import find_child,download_file
            rows=find_child(self.service,folder,name)
            if len(rows)>1: raise ValueError(f'Duplicate persistent file {name}')
            if rows: download_file(self.service,rows[0]['id'],destination)
        elif (folder/name).is_file() and (folder/name).resolve()!=destination.resolve():
            shutil.copy2(folder/name,destination)

    def put(self, folder, path):
        if not folder or not path.is_file(): return
        if self.service:
            from scripts.kaggle_drive import upsert_file
            upsert_file(self.service,folder,path)
        elif (folder/path.name).resolve()!=path.resolve():
            temp=folder/(path.name+'.tmp'); shutil.copy2(path,temp); os.replace(temp,folder/path.name)

    def store(self, folder):
        from repro.persistence import LocalStore,DriveStore
        if not folder: return None,None
        if self.service: return DriveStore(folder,service=self.service),{'backend':'gdrive','folder_id':folder}
        return LocalStore(folder),{'backend':'local','root':str(folder.parent),'run_id':folder.name}


def complete_checkpoint(path, epochs):
    import torch
    state=torch.load(path,map_location='cpu',weights_only=False)
    return bool(state.get('epoch_complete',True) and state['epoch']+1>=epochs)


def ensure_reports(args,remote):
    from repro.persistence import restore
    from scripts.visualize_suite import render
    main=args.output/'main'
    if not (main/'COMPLETED.json').exists(): return
    folder=remote.folder('main')
    for name in ('test_predictions.json','suite-config.yaml'):
        remote.get(folder,name,main/name)
    if not (main/'test_predictions.json').exists(): return
    for name in ('graph_stage16.png','graph_stage32.png','graph_edges.json'):
        remote.get(remote.root,name,args.output/name)
    reference=args.output/'multisigngraph'
    if (reference/'COMPLETED.json').exists():
        remote.get(remote.folder('multisigngraph'),'test_predictions.json',reference/'test_predictions.json')
    if not all((args.output/f'graph_stage{s}.png').exists() for s in (16,32)):
        store,_=remote.store(folder)
        if store: restore(store,main/'best.pt','best')
    render(args.output,args.data_root,args.manifests)
    for name in ('qualitative.json','graph_stage16.png','graph_stage32.png','graph_edges.json'):
        remote.put(remote.root,args.output/name)


def summarize(rows, output, remote):
    result=[]
    for row in rows:
        comparison=output/row['id']/'comparison.json'
        done=output/row['id']/'COMPLETED.json'
        if comparison.exists() and done.exists(): result.append(json.loads(comparison.read_text()))
        else: result.append({'id':row['id'],'source':row['source'],'status':'pending','paper_dev_wer':row['paper_dev_wer'],'paper_test_wer':row['paper_test_wer']})
    path=output/'suite_summary.json'; atomic_json(path,result); remote.put(remote.root,path)
    fields=['id','source','status','paper_dev_wer','measured_dev_wer','dev_delta','paper_test_wer','measured_test_wer','test_delta']
    for split in ('dev','test'):
        for metric in ('del','ins'): fields += [f'paper_{split}_{metric}',f'measured_{split}_{metric}',f'{split}_{metric}_delta']
    path=output/'suite_summary.csv'
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); writer.writeheader(); writer.writerows(result)
    remote.put(remote.root,path)


def execute(args, remote):
    import yaml
    from scripts.run_local import code_hash
    from repro.runtime import sha256
    from repro.persistence import restore
    rows=experiments()
    if args.only:
        requested=args.only.split(','); rows=[r for r in rows if r['id'] in requested]
        if {r['id'] for r in rows}!=set(requested): raise ValueError('Unknown --only experiment ID')
    template=yaml.safe_load((ROOT/'configs_repro/phoenix14t_cslr.yaml').read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    manifests={s:sha256(args.manifests/f'{s}.jsonl') for s in ('train','dev','test')}
    start=time.monotonic()
    for number,row in enumerate(rows,1):
        output=args.output/row['id']; output.mkdir(parents=True,exist_ok=True)
        folder=remote.folder(row['id'])
        for name in ('COMPLETED.json','comparison.json'):
            remote.get(folder,name,output/name)
        cfg={**template,**BASE,**row['changes'],'seed':args.seed,'data_root':str(args.data_root),
             'output':str(output),'epochs':args.epochs,'workers':args.workers,
             'resnet_weights':str(ROOT/'assets/resnet18-f37072fd.pth'),
             'code_sha256':code_hash(),'manifest_sha256':manifests,'suite_experiment':row}
        for s in manifests: cfg[s]=str(args.manifests/f'{s}.jsonl')
        identity=signature(cfg)
        done=output/'COMPLETED.json'
        if done.exists():
            marker=json.loads(done.read_text())
            if marker['signature']!=identity: raise ValueError(f'Completed {row["id"]} uses different code/config/data; choose a new campaign directory.')
            if not (output/'comparison.json').exists(): raise RuntimeError('Completed run is missing comparison')
            print(f'[{number}/{len(rows)}] SKIP completed {row["id"]}',flush=True); continue
        remaining=args.minutes-(time.monotonic()-start)/60 if args.minutes else 0
        if args.minutes and remaining<3:
            print('SESSION_SAVED: session budget reached; Run All again.',flush=True); break
        if shutil.disk_usage(output).free<10*1024**3:
            raise RuntimeError('Less than 10 GiB scratch free: stop before training another model. Existing data/checkpoints retained.')
        if cfg['backbone'] in ('swin_t','pvig_tiny'):
            from scripts.fetch_backbones import ensure
            cfg['backbone_weights']=str(ROOT/'assets'/('swin_t-704ceda3.pth' if cfg['backbone']=='swin_t' else 'pvig_ti_78.5.pth.tar'))
            if not Path(cfg['backbone_weights']).is_file():
                raise FileNotFoundError('Prepare all backbone weights before starting offline suite training.')
        store,persistence=remote.store(folder)
        if persistence: cfg['persistence']=persistence
        cfg['max_session_minutes']=max(1,remaining-2) if args.minutes else 0
        cfg['checkpoint_every_updates']=args.checkpoint_every
        config=output/'suite-config.yaml'
        previous=output/'identity.json'; remote.get(folder,'identity.json',previous)
        if previous.exists() and json.loads(previous.read_text())['signature']!=identity:
            raise ValueError(f'{row["id"]} checkpoint identity differs; use a new campaign output.')
        atomic_json(previous,{'signature':identity}); remote.put(folder,previous)
        config.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding='utf-8')
        if store:
            for kind in ('last','best'):
                destination=output/f'{kind}.pt'
                # Restore latest committed generation, not an unverified file copy.
                restore(store,destination,kind)
        last=output/'last.pt'
        command=[sys.executable,'scripts/launch.py','--config',str(config)]
        if last.exists(): command+=['--resume',str(last)]
        print(f'[{number}/{len(rows)}] TRAIN {row["id"]} ({row["source"]})',flush=True)
        subprocess.run([sys.executable,'scripts/doctor.py','--config',str(config)],cwd=ROOT,check=True)
        process=None
        try:
            with (output/'training.log').open('a',encoding='utf-8') as log:
                process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                with (output/'training.log').open('r',encoding='utf-8',errors='replace') as reader:
                    while process.poll() is None:
                        for line in reader: print(line.rstrip(),flush=True)
                        try: process.wait(timeout=15)
                        except subprocess.TimeoutExpired: pass
                    for line in reader: print(line.rstrip(),flush=True)
            if process.returncode:
                print('\n'.join((output/'training.log').read_text(errors='replace').splitlines()[-30:]),flush=True)
                raise RuntimeError(f'{row["id"]} failed; campaign stopped without changing recipe.')
        except KeyboardInterrupt:
            if process and process.poll() is None:
                process.terminate()
                try: process.wait(timeout=90)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
            raise
        except Exception:
            raise
        finally:
            for name in FILES[:-1]: remote.put(folder,output/name)
        if not last.exists() or not complete_checkpoint(last,args.epochs):
            print('SESSION_SAVED: current experiment is incomplete; rerun to resume.',flush=True); break
        best=output/'best.pt'
        if not best.exists(): raise FileNotFoundError('Training finished without best.pt')
        for split in ('dev','test'):
            subprocess.run([sys.executable,'scripts/launch.py','--config',str(config),'--mode','eval',
                           '--split',split,'--checkpoint',str(best)],cwd=ROOT,check=True)
        comparison={**row,'status':'measured reconstruction'}
        for split in ('dev','test'):
            metric=json.loads((output/f'{split}_metrics.json').read_text())[cfg.get('eval_head','sequence')]
            comparison[f'measured_{split}_wer']=metric['wer']
            comparison[f'measured_{split}_del']=metric['del']; comparison[f'measured_{split}_ins']=metric['ins']
            for kind in ('del','ins'):
                target=row[f'paper_{split}_{kind}']
                comparison[f'{split}_{kind}_delta']=metric[kind]-target if target is not None else None
            target=row[f'paper_{split}_wer']
            comparison[f'{split}_delta']=metric['wer']-target if target is not None else None
        atomic_json(output/'comparison.json',comparison)
        # Publish measurements before marking the experiment completed remotely.
        for name in FILES[:-1]: remote.put(folder,output/name)
        atomic_json(done,{'signature':identity,'status':'measured reconstruction','best_selection':'dev WER','epochs':args.epochs})
        remote.put(folder,done)
        summarize(rows,args.output,remote)
    summarize(rows,args.output,remote)
    ensure_reports(args,remote)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,default=ROOT/'data/phoenix14t')
    p.add_argument('--manifests',type=Path,default=ROOT/'data/manifests/phoenix14t')
    p.add_argument('--output',type=Path,default=ROOT/'runs/phoenix14t_cslr_suite_seed0')
    p.add_argument('--persist-root',type=Path)
    p.add_argument('--drive-folder')
    p.add_argument('--minutes',type=float,default=0)
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--checkpoint-every',type=int,default=200)
    p.add_argument('--only',help='Comma-separated experiment IDs, omitted means all 68 (67 quantitative + 1 qualitative reference)')
    p.add_argument('--plan',action='store_true')
    a=p.parse_args()
    if a.plan:
        print(json.dumps(experiments(),ensure_ascii=False,indent=2)); return
    if a.epochs<1 or a.minutes<0 or a.checkpoint_every<1 or a.workers<0: p.error('Invalid training/session limits')
    if a.persist_root and a.drive_folder: p.error('Choose either mounted Drive or Drive API')
    os.chdir(ROOT)
    def stop(*_): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    remote=Remote(a)
    execute(a,remote)


if __name__=='__main__':
    try: main()
    except KeyboardInterrupt:
        print('SESSION_SAVED: interrupted; resume the same campaign.',flush=True)
        raise SystemExit(130)
