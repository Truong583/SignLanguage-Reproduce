"""Standard-library workspace locks, immutable release paths, and compatibility checks."""
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

DEFAULT_CAMPAIGN='phoenix14t_cslr_suite_seed0'


def state_dir(workspace):
    workspace=Path(workspace).resolve()
    path=workspace/'.updates'
    if not path.resolve().is_relative_to(workspace): raise ValueError('Update directory escapes workspace')
    path.mkdir(exist_ok=True)
    return path


def atomic_json(path,value):
    path=Path(path); temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    path.parent.mkdir(parents=True,exist_ok=True)
    with temporary.open('w',encoding='utf-8') as out:
        json.dump(value,out,indent=2); out.write('\n'); out.flush(); os.fsync(out.fileno())
    os.replace(temporary,path)


class OperationLock:
    def __init__(self,workspace,name='operation.lock'):
        if name not in ('operation.lock','supervisor.lock'): raise ValueError('Invalid lock name')
        self.path=state_dir(workspace)/name
    def __enter__(self):
        self.stream=self.path.open('a+b')
        if self.path.stat().st_size==0: self.stream.write(b'0'); self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            raise RuntimeError('This workspace is running or updating. Stop its run before updating.') from exc
        return self
    def __exit__(self,*args):
        if os.name=='nt':
            import msvcrt
            self.stream.seek(0); msvcrt.locking(self.stream.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(self.stream.fileno(),fcntl.LOCK_UN)
        self.stream.close()


def release_path(workspace,ident):
    if not re.fullmatch(r'[a-f0-9]{40}|[a-f0-9]{64}',ident): raise ValueError('Invalid release identifier')
    root=state_dir(workspace)/'releases'; root.mkdir(exist_ok=True)
    if not root.resolve().is_relative_to(Path(workspace).resolve()): raise ValueError('Release directory escapes workspace')
    path=root/ident
    if not path.resolve().is_relative_to(root.resolve()): raise ValueError('Release escapes managed directory')
    return path


def active_release(workspace):
    path=state_dir(workspace)/'active.json'
    if not path.exists(): return None
    value=json.loads(path.read_text()); candidate=release_path(workspace,value['release_id'])
    if not (candidate/'run.py').is_file() or not (candidate/'BUNDLE_SHA256.json').is_file():
        raise RuntimeError('Active release is incomplete; use update.py --rollback.')
    return candidate


def settings(workspace):
    active=state_dir(workspace)/'active.json'
    if active.exists():
        value=json.loads(active.read_text())
        if 'settings' in value: return value['settings']
    path=state_dir(workspace)/'config.json'
    return json.loads(path.read_text()) if path.exists() else {'repo':None,'branch':'main','campaign':DEFAULT_CAMPAIGN}


def training_hash(root):
    h=hashlib.sha256(); root=Path(root)
    for path in sorted((root/'repro').rglob('*.py')):
        h.update(path.relative_to(root).as_posix().encode()); h.update(path.read_bytes())
    return h.hexdigest()


def recipe_hash(root):
    h=hashlib.sha256(); root=Path(root)
    files=[root/'configs_repro/phoenix14t_cslr.yaml',*sorted((root/'configs_repro/phoenix14t_suite').glob('*.yaml'))]
    for path in files:
        if path.exists(): h.update(path.relative_to(root).as_posix().encode()); h.update(path.read_bytes())
    return h.hexdigest()


def image_name(root,base_image):
    digest=hashlib.sha256((Path(root)/'BUNDLE_SHA256.json').read_bytes()+base_image.encode()).hexdigest()
    return 'mixsigngraph-repro:'+digest[:16]


def workspace_label(workspace): return hashlib.sha256(str(Path(workspace).resolve()).encode()).hexdigest()[:24]


def verify_source(root):
    root=Path(root).resolve()
    files=json.loads((root/'BUNDLE_SHA256.json').read_text(encoding='utf-8'))
    for name,expected in files.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root): raise ValueError('Source manifest escapes release')
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Source checksum mismatch: {name}')
    for required in ('run.py','update.py','Dockerfile','scripts/run_local.py'):
        if required not in files: raise ValueError(f'Release manifest omits required entry: {required}')


def mount_points(root):
    # Nested Docker binds need existing targets inside the readonly source bind.
    root=Path(root).resolve()
    for name in ('data','assets','runs'):
        path=root/name
        if not path.resolve().is_relative_to(root) or path.is_symlink(): raise ValueError('Unsafe source mount point')
        path.mkdir(exist_ok=True)


def valid_campaign(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',value): raise ValueError('Campaign name: letters/numbers/underscore/hyphen only, max 80 characters')
    return value
