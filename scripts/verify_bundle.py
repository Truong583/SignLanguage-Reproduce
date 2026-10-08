"""Write/check SHA-256 inventory. Does not modify or delete project data."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXCLUDE={'.git','.idea','__pycache__','.pytest_cache','.venv','.updates'}

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--write',action='store_true')
    p.add_argument('--include-data',action='store_true')
    p.add_argument('--manifest',default='BUNDLE_SHA256.json')
    a=p.parse_args()
    manifest=ROOT/a.manifest
    if a.write:
        inventory={}
        for path in sorted(ROOT.rglob('*')):
            relative=path.relative_to(ROOT)
            if not path.is_file() or path==manifest or any(part in EXCLUDE for part in relative.parts): continue
            if not a.include_data and relative.parts[0] in {'data','assets','runs'}: continue
            if path.suffix in {'.zip','.tar','.pyc'}: continue
            if path.name=='.env' or path.suffix in {'.pem','.key'} or 'client_secret' in path.name or 'oauth_credentials' in path.name: continue
            inventory[relative.as_posix()]=digest(path)
        manifest.write_text(json.dumps(inventory,indent=2)+'\n',encoding='utf-8')
        print(f'Inventoried {len(inventory)} files; include_data={a.include_data}')
    else:
        inventory=json.loads(manifest.read_text(encoding='utf-8'))
        for name,expected in inventory.items():
            path=(ROOT/name).resolve()
            if not path.is_relative_to(ROOT): raise ValueError('Inventory path escapes root')
            if not path.is_file() or digest(path)!=expected: raise ValueError(f'Missing or changed: {name}')
        print(f'PASS: {len(inventory)} file checksums match')
        if (ROOT/'RELEASE.json').exists():
            release=json.loads((ROOT/'RELEASE.json').read_text(encoding='utf-8'))
            print(f'Release: {release["version"]}; {release["experiments"]} experiment runs')

if __name__=='__main__': main()
