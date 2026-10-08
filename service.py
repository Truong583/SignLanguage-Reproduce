"""Install a scoped Linux service running as the current user, never as root."""
import argparse
import getpass
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from scripts.deployment import OperationLock,state_dir,workspace_label,verify_source,atomic_json

ROOT=Path(__file__).resolve().parent


def service_name(root): return 'signlanguage-'+workspace_label(root)[:12]+'.service'


def quote(value):
    return '"'+str(value).replace('\\','\\\\').replace('"','\\"').replace('%','%%').replace('\n','\\n')+'"'


def unit(root,python,user,suite=False):
    if not user or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in user):
        raise ValueError('Unsupported account name')
    mode='--suite' if suite else '--single'
    return '\n'.join(['[Unit]','Description=SignLanguage supervised PHOENIX14T run',
      'Requires=docker.service','After=docker.service network-online.target','Wants=network-online.target',
      'RequiresMountsFor='+quote(root),
      '[Service]','Type=simple','User='+user,'Group=docker','WorkingDirectory='+quote(root),
      'ExecStart='+quote(python)+' '+quote(Path(root)/'service.py')+' _run '+mode,
      'Environment=PYTHONUNBUFFERED=1','Restart=on-failure','RestartSec=30',
      'KillMode=mixed','TimeoutStopSec=180','UMask=0077',
      '[Install]','WantedBy=multi-user.target',''])


def daemon(suite):
    # A cold boot may start this service before the Docker API is ready.
    while True:
        try:
            probe=subprocess.run(['docker','info','--format','{{.OSType}}'],capture_output=True,text=True,timeout=20)
            if probe.returncode==0 and probe.stdout.strip()=='linux': break
        except (OSError,subprocess.TimeoutExpired): pass
        print('Waiting for Docker before resuming the saved workspace...',flush=True)
        time.sleep(5)
    args=[sys.executable,str(ROOT/'run.py')]
    if not suite: args.append('--single')
    os.execv(sys.executable,args)


def main():
    parser=argparse.ArgumentParser(description='Scoped Linux background service; defaults to the main CSLR run.')
    parser.add_argument('action',choices=['install','stop','status','logs','_run'])
    mode=parser.add_mutually_exclusive_group(); mode.add_argument('--single',action='store_true'); mode.add_argument('--suite',action='store_true')
    args=parser.parse_args()
    if sys.platform!='linux': raise RuntimeError('Run this command on the Linux training machine.')
    if args.action=='_run': daemon(args.suite); return
    name=service_name(ROOT)
    if args.action!='install':
        command=['systemctl','status',name,'--no-pager'] if args.action=='status' else ['sudo','journalctl','-u',name,'-f'] if args.action=='logs' else ['sudo','systemctl','stop',name]
        return subprocess.call(command)
    if os.getuid()==0: raise RuntimeError('Use python3 service.py install as your regular account, not sudo python.')
    import pwd
    user=pwd.getpwuid(os.getuid()).pw_name
    if not shutil.which('docker') or not shutil.which('systemctl'): raise RuntimeError('Docker and systemd are required.')
    root=state_dir(ROOT)
    if not (root/'machine.json').exists() or not (root/'wandb_key').exists(): raise RuntimeError('Configure setup_machine.py first.')
    with OperationLock(ROOT,'supervisor.lock'),OperationLock(ROOT):
        verify_source(ROOT)
        source=root/'system-service.service'
        source.write_text(unit(ROOT,Path(sys.executable).resolve(),user,args.suite),encoding='utf-8')
        print('Installing '+name+' as '+user+' for '+('68-run suite' if args.suite else 'main PHOENIX14T CSLR run'),flush=True)
        subprocess.run(['sudo','install','-m','0644',str(source),'/etc/systemd/system/'+name],check=True)
        subprocess.run(['sudo','systemctl','daemon-reload'],check=True)
        subprocess.run(['sudo','systemctl','enable',name],check=True)
        atomic_json(root/'service.json',{'unit':name,'user':user,'suite':args.suite})
    subprocess.run(['sudo','systemctl','start',name],check=True)
    print('Service started and enabled at boot. Status: python3 service.py status')
    print('Logs: python3 service.py logs. Stop cleanly: python3 service.py stop')
    return 0


if __name__=='__main__':
    try: raise SystemExit(main())
    except (RuntimeError,ValueError,subprocess.CalledProcessError) as error:
        print('SERVICE STOPPED: '+str(error),file=sys.stderr); raise SystemExit(1)
