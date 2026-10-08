import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',required=True)
    p.add_argument('--dev',type=Path,required=True)
    p.add_argument('--test',type=Path,required=True)
    p.add_argument('--head',choices=['sequence','conv'],default='sequence')
    a=p.parse_args()
    targets=json.loads((ROOT/'docs/paper_targets.json').read_text())
    target=targets[a.experiment]
    output={'experiment':a.experiment,'status':'Measured reconstruction; no claim of exact reproduction','metrics':{}}
    for split,path in [('dev',a.dev),('test',a.test)]:
        result=json.loads(path.read_text())
        for metric,expected in target[split].items():
            actual=result[a.head]['wer'] if metric=='wer' else result[metric]
            output['metrics'][f'{split}_{metric}']={'paper':expected,'measured':actual,'delta':actual-expected}
    print(json.dumps(output,indent=2))

if __name__=='__main__': main()
