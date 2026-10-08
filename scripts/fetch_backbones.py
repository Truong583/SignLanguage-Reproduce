"""Fetch official ImageNet backbones, record source and SHA256, validate with strict load."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SOURCES={
    'swin_t': ('swin_t-704ceda3.pth','https://download.pytorch.org/models/swin_t-704ceda3.pth','704ceda373461b0a224fcdddd75cd2a5e9f8064512ed47adbddef7f343fd147b'),
    'pvig_tiny': ('pvig_ti_78.5.pth.tar','https://github.com/huawei-noah/Efficient-AI-Backbones/releases/download/pyramid-vig/pvig_ti_78.5.pth.tar','06c49bda678b26a8cbc69c69e2762c3496eb4d9db8cacb5e88cdd517f56e4d4f'),
}


def ensure(kind, output):
    filename,url,prefix=SOURCES[kind]; output=Path(output); output.mkdir(parents=True,exist_ok=True)
    path=output/filename
    if not path.exists():
        temp=path.with_name(filename+'.part')
        print(f'Downloading official {kind} weights...',flush=True)
        urllib.request.urlretrieve(url,temp)
        temp.replace(path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if prefix and not digest.startswith(prefix): raise ValueError(f'Checksum mismatch: {path}')
    provenance=output/(filename+'.source.json')
    if provenance.exists() and json.loads(provenance.read_text())['sha256']!=digest:
        raise ValueError('Cached backbone hash changed')
    import torch
    from repro.ablation import SwinFrames,PVIGTiny
    state=torch.load(path,map_location='cpu',weights_only=True)
    state=state.get('state_dict',state.get('model',state))
    state={k.removeprefix('module.'):v for k,v in state.items()}
    model=SwinFrames().model if kind=='swin_t' else PVIGTiny()
    model.load_state_dict(state,strict=True)
    provenance.write_text(json.dumps({'url':url,'sha256':digest,'strict_state_dict_validation':True,
        'pvig_reference_commit':'a328aa023d4e3e66cca690ecc54f77daf436b9ff',
        'note':'PyViG size and head attachment are reconstruction choices; not specified in MixSignGraph release.'},indent=2))
    return path


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--backbone',choices=list(SOURCES),required=True)
    p.add_argument('--output',default='assets'); a=p.parse_args(); ensure(a.backbone,a.output)
