"""Create a source-only portable ZIP from the verified bundle inventory."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
target=ROOT.parent/'SignLanguage-Reproduce-portable.zip'
if target.exists(): raise FileExistsError(f'Refusing to overwrite {target}')
inventory=json.loads((ROOT/'BUNDLE_SHA256.json').read_text(encoding='utf-8'))
with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
    for name,expected in inventory.items():
        path=ROOT/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected: raise ValueError(f'Changed after inventory: {name}')
        archive.write(path,f'{ROOT.name}/{name}')
    archive.write(ROOT/'BUNDLE_SHA256.json',f'{ROOT.name}/BUNDLE_SHA256.json')
    for folder in ['data','assets','runs']:
        archive.writestr(f'{ROOT.name}/{folder}/.gitkeep','')
with zipfile.ZipFile(target) as archive:
    if archive.testzip() is not None: raise RuntimeError('ZIP CRC test failed')
digest=hashlib.sha256(target.read_bytes()).hexdigest()
target.with_suffix('.zip.sha256').write_text(f'{digest}  {target.name}\n')
print(f'{target}\n{target.stat().st_size} bytes\nSHA256 {digest}')
