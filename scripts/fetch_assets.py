"""Explicit online preparation; training itself never downloads anything."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--resnet',action='store_true')
    p.add_argument('--mbart',action='store_true')
    p.add_argument('--mbart-revision',help='Pinned 40-character Hugging Face commit')
    p.add_argument('--output',type=Path,default=Path('assets'))
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if a.resnet:
        path=a.output/'resnet18-f37072fd.pth'
        if not path.exists():
            temporary=path.with_suffix('.download')
            urllib.request.urlretrieve('https://download.pytorch.org/models/resnet18-f37072fd.pth',temporary)
            digest=hashlib.sha256(temporary.read_bytes()).hexdigest()
            if not digest.startswith('f37072fd'): raise ValueError('ResNet checksum mismatch')
            temporary.replace(path)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if not digest.startswith('f37072fd'): raise ValueError('Existing ResNet checksum mismatch')
        (a.output/'resnet.sha256').write_text(f'{digest}  {path.name}\n')
    if a.mbart:
        if not a.mbart_revision or len(a.mbart_revision)!=40 or any(c not in '0123456789abcdef' for c in a.mbart_revision):
            p.error('--mbart requires a full pinned --mbart-revision')
        from huggingface_hub import snapshot_download
        snapshot_download('facebook/mbart-large-cc25',revision=a.mbart_revision,
                          local_dir=a.output/'mbart-large-cc25',
                          allow_patterns=['*.json','*.model','*.bin','*.safetensors'])
        (a.output/'mbart-source.json').write_text(json.dumps({'repo':'facebook/mbart-large-cc25','revision':a.mbart_revision},indent=2))
    if not a.resnet and not a.mbart: p.error('Choose --resnet and/or --mbart')

if __name__=='__main__': main()
