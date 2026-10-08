"""Read immutable per-video ZIPs from Drive with a bounded, rank-local cache."""
import os
from pathlib import Path
import shutil
import time
from .persistence import DriveStore, digest


class ClipCache:
    def __init__(self, store=None, root=None, limit=None):
        self.store=store or DriveStore(os.environ['SIGN_DRIVE_ROOT'])
        self.root=Path(root or os.environ.get('SIGN_DRIVE_CACHE','/tmp/signlanguage-cache'))/os.environ.get('RANK','0')
        self.root.mkdir(parents=True,exist_ok=True)
        self.limit=int(limit if limit is not None else float(os.environ.get('SIGN_DRIVE_CACHE_GB','2'))*1024**3)

    def fetch(self,row):
        expected=row['clip_sha256']
        if len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected): raise ValueError('Invalid clip checksum')
        size=int(row['clip_bytes'])
        if size<=0 or size>self.limit: raise ValueError('One video exceeds cache budget; increase CACHE_GB')
        target=self.root/(expected+'.zip')
        if target.exists() and target.stat().st_size==size and digest(target)==expected:
            os.utime(target,None)
            return target
        target.unlink(missing_ok=True)
        files=sorted(self.root.glob('*.zip'),key=lambda p:p.stat().st_mtime)
        used=sum(p.stat().st_size for p in files)
        for path in files:
            if used+size<=self.limit: break
            used-=path.stat().st_size
            path.unlink()
        if shutil.disk_usage(self.root).free<size+128*1024**2: raise IOError('Insufficient temporary space for one video')
        temp=target.with_suffix('.partial')
        try:
            self.store.get(row['clip_id'],temp)
            if temp.stat().st_size!=size or digest(temp)!=expected: raise IOError('Drive video checksum mismatch')
            os.replace(temp,target)
        finally: temp.unlink(missing_ok=True)
        return target


_cache=None
def clip_path(row):
    global _cache
    if _cache is None: _cache=ClipCache()
    return _cache.fetch(row)
