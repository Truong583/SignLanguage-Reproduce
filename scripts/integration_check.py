"""Exercise train -> checkpoint -> resume -> eval on tiny synthetic FEATURES.

This does not validate dataset results or the RGB backbone (see repro.smoke).
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--directory',default='runs/integration')
    p.add_argument('--distributed',action='store_true',help='Two CPU ranks, requires Linux Gloo support')
    a=p.parse_args()
    root=Path(a.directory).resolve()
    root.mkdir(parents=True,exist_ok=False)
    data=root/'data'
    data.mkdir()
    cfg={'task':'cslr','device':'cpu','input_kind':'features','feature_dim':16,'hidden_size':32,
         'data_root':str(data),'output':str(root/'run'),'global_batch':4,'micro_batch':1,
         'eval_batch':1,'workers':0,'epochs':1,'precision':'fp32','seed':0,'distillation':1.,'ctc_beam':2}
    for split,count in [('train',8),('dev',3),('test',3)]:
        rows=[]
        for i in range(count):
            name=f'{split}_{i}'
            np.save(data/f'{name}.npy',np.random.default_rng(i).normal(size=(12,16)).astype('float32'))
            rows.append({'id':name,'features':f'{name}.npy','gloss':'a b','text':''})
        manifest=data/f'{split}.jsonl'
        manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        cfg[split]=str(manifest)
    config=root/'config.yaml'
    config.write_text(yaml.safe_dump(cfg))
    env=os.environ.copy()
    prefix=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','-m','repro.train'] if a.distributed else [sys.executable,'-m','repro.train']
    def run(*extra):
        subprocess.run([*prefix,'--config',str(config),*extra],env=env,cwd=ROOT,check=True)
    run()
    cfg['epochs']=2
    config.write_text(yaml.safe_dump(cfg))
    run('--resume',str(root/'run/last.pt'))
    run('--mode','eval','--checkpoint',str(root/'run/best.pt'),'--split','test')
    metrics=json.loads((root/'run/test_metrics.json').read_text())
    assert metrics['samples']==3
    history=(root/'run/history.jsonl').read_text().splitlines()
    assert [json.loads(r)['epoch'] for r in history]==[1,2]
    print('Integration PASS: checkpoint, resume, complete evaluation. Synthetic data only.')

if __name__=='__main__': main()
