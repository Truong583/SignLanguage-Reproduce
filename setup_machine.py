"""One-time Linux setup. Tokens are entered locally, never committed or put in diagnostics."""
import argparse
import getpass
import json
import os
from pathlib import Path
import sys
import urllib.request
import urllib.error
from scripts.deployment import state_dir,atomic_json,valid_campaign
from scripts.privacy import require_private

ROOT=Path(__file__).resolve().parent


def secret(path,value):
    if not value: raise ValueError('Empty secret')
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    with os.fdopen(fd,'w') as out: out.write(value+'\n')
    os.chmod(path,0o600)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--entity'); p.add_argument('--project',default='signlanguage-reproduction')
    a=p.parse_args(); root=state_dir(ROOT)
    if os.name!='nt': os.chmod(root,0o700)
    entity=a.entity or input('W&B account/team name: ').strip()
    if not entity: raise ValueError('W&B entity is required')
    request=urllib.request.Request('https://api.github.com/repos/Truong583/SignLanguage-Reproduce',
        headers={'Accept':'application/vnd.github+json'})
    token=None
    try:
        with urllib.request.urlopen(request,timeout=30) as response: repo=json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code!=404: raise
        token=getpass.getpass('Private GitHub repo: fine-grained token (Contents Read-only): ').strip()
        request.add_header('Authorization','Bearer '+token)
        with urllib.request.urlopen(request,timeout=30) as response: repo=json.load(response)
    public=not repo['private']
    if not public and not token: raise RuntimeError('Private repository requires a scoped token.')
    key=getpass.getpass('W&B API key for the dedicated private research account/project: ').strip()
    require_private(entity,a.project,key)
    if token: secret(root/'github_token',token)
    secret(root/'wandb_key',key)
    machine={'schema':1,'repo':'https://github.com/Truong583/SignLanguage-Reproduce.git','branch':'main',
      'wandb_entity':entity,'wandb_project':a.project,'poll_seconds':300,'public_repo':public,
      'base_image':'pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime'}
    atomic_json(root/'machine.json',machine)
    atomic_json(root/'config.json',{'repo':machine['repo'],'branch':'main','campaign':'phoenix14t_cslr_suite_seed0'})
    print('Configured. Main experiment: python3 run.py --single. Optional boot/background mode: python3 service.py install --single')
    print('No SSH server, root installation or global Git credential configuration is needed.')
    print('Dashboard: https://wandb.ai/'+entity+'/'+a.project)


if __name__=='__main__': main()
