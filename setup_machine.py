"""One-time Linux setup. Tokens are entered locally, never committed or put in diagnostics."""
import argparse
import getpass
import json
import os
from pathlib import Path
import sys
import urllib.request
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
    token=getpass.getpass('GitHub fine-grained token (only this repo; Contents Read-only): ').strip()
    request=urllib.request.Request('https://api.github.com/repos/Truong583/SignLanguage-Reproduce',
        headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json'})
    repo=json.load(urllib.request.urlopen(request,timeout=30))
    if not repo['private']: raise RuntimeError('Repository is not Private; setup stopped.')
    key=getpass.getpass('W&B API key for the dedicated private research account/project: ').strip()
    require_private(entity,a.project,key)
    secret(root/'github_token',token); secret(root/'wandb_key',key)
    machine={'schema':1,'repo':'https://github.com/Truong583/SignLanguage-Reproduce.git','branch':'main',
      'wandb_entity':entity,'wandb_project':a.project,'poll_seconds':300,
      'base_image':'pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime'}
    atomic_json(root/'machine.json',machine)
    atomic_json(root/'config.json',{'repo':machine['repo'],'branch':'main','campaign':'phoenix14t_cslr_suite_seed0'})
    python=str(Path(sys.executable).resolve())
    def quote(value): return '"'+value.replace('\\','\\\\').replace('"','\\"').replace('%','%%')+'"'
    service='\n'.join(['[Unit]','Description=SignLanguage reproduction supervisor','After=network-online.target',
      '[Service]','Type=simple','WorkingDirectory='+quote(str(ROOT)),
      'ExecStart='+quote(python)+' '+quote(str(ROOT/'run.py')),'Restart=on-failure','RestartSec=30',
      'Environment=PYTHONUNBUFFERED=1','[Install]','WantedBy=default.target',''])
    (root/'signlanguage.service').write_text(service)
    print('Configured. Run python run.py on this Linux machine. Optional user service unit: .updates/signlanguage.service')
    print('No SSH server, root installation or global Git credential configuration is needed.')
    print('Dashboard: https://wandb.ai/'+entity+'/'+a.project)


if __name__=='__main__': main()
