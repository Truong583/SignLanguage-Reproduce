"""Laptop command: publish source-only files to the verified private GitHub repository."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
from scripts.deployment import OperationLock,state_dir
from scripts.diagnostics import redact

ROOT=Path(__file__).resolve().parent
REPO='https://github.com/Truong583/SignLanguage-Reproduce.git'


def command(*args,cwd=None):
    result=subprocess.run(['git',*args],cwd=cwd,capture_output=True,text=True)
    if result.returncode: raise RuntimeError(redact(result.stderr.strip()))
    return result.stdout.strip()


def main():
    # Uses the laptop's existing credential manager; never copies credentials to a server/repository.
    result=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',
        capture_output=True,text=True,check=False)
    credentials=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
    token=credentials.get('password')
    if not token: raise RuntimeError('Sign into GitHub in the laptop Git credential manager first.')
    request=urllib.request.Request('https://api.github.com/repos/Truong583/SignLanguage-Reproduce',
        headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json'})
    with urllib.request.urlopen(request,timeout=30) as response: repo=json.load(response)
    if not repo.get('private'): raise RuntimeError('Publish stopped: target repository must be Private.')
    with OperationLock(ROOT):
        subprocess.run([sys.executable,'scripts/verify_bundle.py','--write'],cwd=ROOT,check=True)
        inventory=json.loads((ROOT/'BUNDLE_SHA256.json').read_text())
        target=state_dir(ROOT)/'publisher'; target.mkdir(exist_ok=True)
        if not target.resolve().is_relative_to(state_dir(ROOT).resolve()): raise ValueError('Invalid publish checkout')
        if not (target/'.git').exists(): command('init','-b','main',cwd=target)
        refs=command('ls-remote',REPO,'refs/heads/main',cwd=target)
        if refs:
            command('fetch','--depth=1',REPO,'main',cwd=target)
            command('checkout','-B','main','FETCH_HEAD',cwd=target)
        # Only files tracked in this dedicated checkout are removed, never the working project/data.
        tracked=command('ls-files','-z',cwd=target).split('\0')
        for name in tracked:
            if not name: continue
            path=target/name
            if not path.resolve().is_relative_to(target.resolve()): raise ValueError('Invalid tracked publish path')
            path.unlink(missing_ok=True)
        for name in [*inventory,'BUNDLE_SHA256.json']:
            source=ROOT/name
            if name in inventory and hashlib.sha256(source.read_bytes()).hexdigest()!=inventory[name]:
                raise ValueError('Source changed during publish')
            dest=target/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,dest)
        command('add','-A',cwd=target)
        if not command('status','--porcelain',cwd=target):
            print('Private repository already has this source.'); return
        command('-c','user.name=Truong583','-c','user.email=Truong583@users.noreply.github.com',
            'commit','-m','Update PHOENIX14T CSLR reproduction and supervised deployment',cwd=target)
        command('push',REPO,'HEAD:refs/heads/main',cwd=target)
        print('Published source to private repository: '+REPO)
        print('Revision: '+command('rev-parse','HEAD',cwd=target))


if __name__=='__main__':
    try: main()
    except Exception as exc: print('PUBLISH STOPPED: '+redact(str(exc)),file=sys.stderr); raise SystemExit(1)
