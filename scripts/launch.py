"""Launch only the visible GPUs; preserve global batch without LR rescaling."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--gpus',type=int,help='Count within CUDA_VISIBLE_DEVICES; auto uses a divisor of global batch')
    p.add_argument('--micro-batch',type=int)
    p.add_argument('--plan',action='store_true')
    a,rest=p.parse_known_args()
    import torch,yaml,json
    from repro.runtime import batch_plan
    cfg=yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    count=torch.cuda.device_count()
    if not count: raise SystemExit('Training launcher requires CUDA. Use python -m repro.smoke --device cpu for CPU tests.')
    plan=batch_plan(count,cfg.get('global_batch',6),a.micro_batch or cfg.get('micro_batch',1),a.gpus)
    print(json.dumps(plan),flush=True)
    if a.plan: return
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1',
             f'--nproc_per_node={plan["world_size"]}','-m','repro.train','--config',a.config,
             '--micro-batch',str(plan['micro_batch']),*rest]
    from scripts.resource_diagnostics import memory_snapshot, exit_evidence
    before=memory_snapshot()
    print('Container memory: '+json.dumps(before),flush=True)
    result=subprocess.call(command,cwd=ROOT)
    if result:
        print('RESOURCE_EXIT: '+json.dumps(exit_evidence(result,before,memory_snapshot())),flush=True)
    raise SystemExit(result)

if __name__=='__main__': main()
