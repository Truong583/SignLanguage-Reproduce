import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--full',action='store_true',help='Open every frame, slower but detects corruption')
    a=p.parse_args()
    import yaml,torch
    from repro.runtime import environment,batch_plan
    cfg=yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    print(json.dumps(environment(),indent=2))
    if torch.cuda.is_available(): print(json.dumps(batch_plan(torch.cuda.device_count(),cfg.get('global_batch',6),cfg.get('micro_batch',1))))
    summary=check_data(cfg,a.full)
    print(json.dumps(summary,indent=2))
    print('Preflight passed. RGB training resampling enforces CTC length; evaluation frames and references are unchanged.',flush=True)


def check_data(cfg,full=False):
    from PIL import Image
    from repro.data import read_manifest,frame_paths,safe_path,build_vocab,encode_targets
    sets={s:read_manifest(cfg[s]) for s in ['train','dev','test']}
    vocab=build_vocab(sets['train'],cfg.get('target_field','gloss'))
    seen=set()
    summary={}
    for split,rows in sets.items():
        unknown=0
        short=[]
        maximum_frames=0
        maximum_padded_frames=0
        for row in rows:
            if row['id'] in seen: raise ValueError(f'Split overlap: {row["id"]}')
            seen.add(row['id'])
            if cfg.get('input_kind','rgb')=='rgb':
                paths=frame_paths(cfg['data_root'],row)
                if not paths: raise ValueError(f'No frames: {row["id"]}')
                if full:
                    for path in paths:
                        with Image.open(path) as image: image.verify()
                length=len(paths)
            else:
                import numpy as np
                arr=np.load(safe_path(cfg['data_root'],row['features']),mmap_mode='r',allow_pickle=False)
                if arr.ndim!=2 or len(arr)<1 or arr.shape[1]!=cfg['feature_dim'] or not np.isfinite(arr).all():
                    raise ValueError(f'Invalid features: {row["id"]}')
                length=len(arr)
            labels,_,required=encode_targets([row],vocab,cfg.get('target_field','gloss'))
            unknown+=int((labels==1).sum())
            needed=int(required[0]); steps=(length+3)//4
            maximum_frames=max(maximum_frames,length)
            upper=max(int(length*1.2),4*needed-3) if split=='train' and cfg.get('input_kind','rgb')=='rgb' else length
            maximum_padded_frames=max(maximum_padded_frames,((upper+3)//4)*4+12)
            if needed>steps:
                detail={'id':row['id'],'source_frames':length,'source_ctc_steps':steps,
                        'required_ctc_steps':needed}
                if split=='train':
                    if cfg.get('input_kind','rgb')!='rgb':
                        raise ValueError(f'Impossible CTC alignment: {row["id"]}; split={split}, source_frames={length}, ctc_steps={steps}, required={needed}. Feature inputs have no temporal resampling.')
                    # SignDataset already applies max(augmented_length,4*required-3)
                    # and repeats existing frames. Check that actual train policy,
                    # rather than rejecting an unaugmented training sequence.
                    detail['minimum_training_frames']=4*needed-3
                # evaluate() computes WER from predictions, without CTC loss.
                # Short dev/test inputs are reported, never resampled using labels.
                short.append(detail)
            if cfg.get('task')=='slt' and not row.get('text'): raise ValueError(f'Missing translation: {row["id"]}')
        summary[split]={'samples':len(rows),'unknown_target_tokens':unknown,
                        'maximum_source_frames':maximum_frames,
                        'maximum_padded_frames_upper_bound':maximum_padded_frames,
                        'ctc_length_adjusted_samples':len(short) if split=='train' else 0,
                        'decode_length_limited_samples':len(short) if split!='train' else 0,
                        'short_sequence_examples':short[:5]}
    if cfg.get('input_kind','rgb')=='rgb':
        key='backbone_weights' if cfg.get('backbone') in ('swin_t','pvig_tiny') else 'resnet_weights'
        if not Path(cfg[key]).is_file(): raise FileNotFoundError(cfg[key])
    if cfg.get('task')=='slt' and not Path(cfg['mbart_path']).is_dir(): raise FileNotFoundError(cfg['mbart_path'])
    return summary

if __name__=='__main__': main()
