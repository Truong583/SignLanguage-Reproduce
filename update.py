"""Fetch a pinned private-repository revision, validate, then atomically activate.

Does not git-pull the live code or overwrite data/assets/runs. No global Git configuration.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import shlex
import urllib.request
import uuid
import base64

from scripts.deployment import (OperationLock,state_dir,release_path,active_release,settings,
    atomic_json,training_hash,recipe_hash,valid_campaign,verify_source,mount_points,image_name,workspace_label)
from scripts.diagnostics import Recorder,redact

SOURCE=Path(__file__).resolve().parent
WORKSPACE=Path(os.environ.get('SIGNLANGUAGE_WORKSPACE',SOURCE)).resolve()
DEFAULT_REPO='https://github.com/Truong583/SignLanguage-Reproduce.git'
BASE_IMAGE='pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime'


def repository(value):
    if re.fullmatch(r'https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?',value): return value
    if re.fullmatch(r'git@github\.com:[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?',value): return value
    if Path(value).is_dir(): return str(Path(value).resolve())  # local repository tests/offline deployment
    raise ValueError('Use a credential-free GitHub HTTPS/SSH URL or an existing local repository path.')


def git_env():
    env=os.environ.copy(); env['GIT_TERMINAL_PROMPT']='0'; env['GCM_INTERACTIVE']='never'
    for name in list(env):
        if name.startswith('GIT_TRACE') or name=='GIT_CURL_VERBOSE': del env[name]
    key=state_dir(WORKSPACE)/'deploy_key'; hosts=state_dir(WORKSPACE)/'known_hosts'
    if key.is_file() and hosts.is_file():
        env['GIT_SSH_COMMAND']=shlex.join(['ssh','-i',str(key),'-o','IdentitiesOnly=yes',
            '-o','StrictHostKeyChecking=yes','-o','BatchMode=yes','-o','UserKnownHostsFile='+str(hosts)])
    token=state_dir(WORKSPACE)/'github_token'
    machine=state_dir(WORKSPACE)/'machine.json'
    public=machine.exists() and json.loads(machine.read_text()).get('public_repo',False)
    if token.exists() and not public:
        value=token.read_text().strip()
        encoded=base64.b64encode(('x-access-token:'+value).encode()).decode()
        env['GIT_CONFIG_COUNT']='1'; env['GIT_CONFIG_KEY_0']='http.https://github.com/.extraheader'
        env['GIT_CONFIG_VALUE_0']='AUTHORIZATION: basic '+encoded
    return env


def setup_key(workspace):
    root=state_dir(workspace); key=root/'deploy_key'
    if not key.exists():
        subprocess.run(['ssh-keygen','-t','ed25519','-f',str(key),'-N','','-C','signlanguage-read-only-server'],check=True)
    public=key.with_name('deploy_key.pub')
    if not public.is_file(): raise RuntimeError('Existing deploy key has no public file; do not overwrite it automatically.')
    os.chmod(key,0o600)
    # Official keys over certificate-verified HTTPS; never disable SSH host-key checks.
    request=urllib.request.Request('https://api.github.com/meta',headers={'User-Agent':'SignLanguage-Reproduction'})
    meta=json.load(urllib.request.urlopen(request,timeout=30))
    (root/'known_hosts').write_text(''.join('github.com '+value+'\n' for value in meta['ssh_keys']),encoding='utf-8')
    cfg=settings(workspace); cfg['repo']='git@github.com:Truong583/SignLanguage-Reproduce.git'
    cfg['branch']='main'
    active=root/'active.json'
    if active.exists():
        value=json.loads(active.read_text()); value['settings']=cfg; atomic_json(active,value)
    else: atomic_json(root/'config.json',cfg)
    print('Add this PUBLIC key to GitHub repository Settings > Deploy keys. Leave Allow write access OFF:')
    print(public.read_text().strip())
    print('Private key stays inside this workspace; it is not in source inventory or diagnostic bundles.')


def git(*args):
    result=subprocess.run(['git',*map(str,args)],capture_output=True,text=True,env=git_env())
    if result.returncode: raise RuntimeError('Git failed: '+redact(result.stderr.strip()))
    return result.stdout.strip()


def snapshot(repo_dir,commit,workspace):
    target=release_path(workspace,commit)
    if target.exists(): verify_source(target); mount_points(target); return target
    root=state_dir(workspace); temp=root/('snapshot-'+uuid.uuid4().hex); temp.mkdir()
    archive=root/(temp.name+'.tar')
    try:
        git('--git-dir',repo_dir,'archive','--format=tar','--output',archive,commit)
        with tarfile.open(archive,'r:') as source:
            for member in source:
                name=PurePosixPath(member.name)
                if name.is_absolute() or '..' in name.parts or not (member.isfile() or member.isdir()):
                    raise ValueError('Git release contains unsafe paths, links or special files')
                if name.parts and name.parts[0] in ('.git','.updates'): raise ValueError('Release contains runtime/Git metadata')
                source.extract(member,temp,filter='data')
        verify_source(temp); mount_points(temp); os.replace(temp,target)
    finally:
        archive.unlink(missing_ok=True)
        if temp.exists():
            if not temp.resolve().is_relative_to(root.resolve()): raise ValueError('Unsafe cleanup path')
            shutil.rmtree(temp)
    return target


def bootstrap_snapshot(workspace):
    verify_source(SOURCE)
    ident=hashlib.sha256((SOURCE/'BUNDLE_SHA256.json').read_bytes()).hexdigest()
    target=release_path(workspace,ident)
    if not target.exists():
        temp=state_dir(workspace)/('snapshot-'+uuid.uuid4().hex); temp.mkdir()
        files=json.loads((SOURCE/'BUNDLE_SHA256.json').read_text())
        for name in [*files,'BUNDLE_SHA256.json']:
            destination=temp/name; destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(SOURCE/name,destination)
        verify_source(temp); mount_points(temp); os.replace(temp,target)
    mount_points(target)
    return {'release_id':ident,'commit':None,'origin':'initial portable source'}


def compatibility(old,new,workspace,new_campaign=None):
    changed=training_hash(old)!=training_hash(new) or recipe_hash(old)!=recipe_hash(new)
    runs=Path(workspace)/'runs'
    existing=runs.exists() and (next(runs.rglob('last.pt'),None) is not None or next(runs.rglob('identity.json'),None) is not None)
    if changed and existing and not new_campaign:
        raise RuntimeError('Training code changed and existing runs are present. Update not activated. Review the fix, then use --new-campaign NAME if new training is required. Checkpoint hash guards are retained.')
    if new_campaign:
        valid_campaign(new_campaign); path=runs/new_campaign
        if path.exists() and any(path.iterdir()): raise ValueError('New campaign must be an unused output directory')
    return {'training_code_changed':changed,'existing_runs':existing,
            'resume':'unchanged training fingerprint; runtime config/data guards still apply' if not changed else 'new training revision; do not claim exact continuation of old results'}


def syntax_check(source):
    files=[source/'run.py',source/'update.py',*sorted((source/'scripts').rglob('*.py')),*sorted((source/'repro').rglob('*.py'))]
    for path in files: compile(path.read_text(encoding='utf-8-sig'),str(path),'exec')


def activate(workspace,new,old,cfg):
    root=state_dir(workspace)
    atomic_json(root/'previous.json',old)
    atomic_json(root/'active.json',{**new,'settings':cfg})


def main():
    p=argparse.ArgumentParser(description='Update only when this workspace is stopped; preserve all datasets/checkpoints.')
    p.add_argument('--repo'); p.add_argument('--branch'); p.add_argument('--revision',help='Exact 40/64-char commit; fetched branch HEAD must match')
    p.add_argument('--new-campaign'); p.add_argument('--rollback',action='store_true')
    p.add_argument('--setup-key',action='store_true',help='One-time dedicated SSH key and verified known_hosts; prints public key only')
    p.add_argument('--if-new',action='store_true',help='No-op for an already active or rejected revision')
    p.add_argument('--automatic',action='store_true',help='Use the candidate DEPLOYMENT_POLICY.json for explicit new-campaign decisions')
    p.add_argument('--base-image',default=BASE_IMAGE)
    args=p.parse_args()
    if args.setup_key:
        with OperationLock(WORKSPACE): setup_key(WORKSPACE)
        return
    if shutil.which('git') is None or shutil.which('docker') is None: raise RuntimeError('Git and running Docker are required.')
    with OperationLock(WORKSPACE):
        recorder=Recorder(WORKSPACE,SOURCE)
        try:
            root=state_dir(WORKSPACE); cfg=settings(WORKSPACE)
            running=subprocess.run(['docker','ps','--filter','label=signlanguage.workspace='+workspace_label(WORKSPACE),'--format','{{.ID}}'],check=True,capture_output=True,text=True)
            if running.stdout.strip(): raise RuntimeError('A container for this workspace is still running; update was blocked.')
            current=active_release(WORKSPACE) or SOURCE
            old=json.loads((root/'active.json').read_text()) if (root/'active.json').exists() else bootstrap_snapshot(WORKSPACE)
            if args.rollback:
                if not (root/'previous.json').exists(): raise RuntimeError('No previous managed release exists.')
                new=json.loads((root/'previous.json').read_text()); candidate=release_path(WORKSPACE,new['release_id'])
                cfg=new.get('settings',cfg)
            else:
                repo=repository(args.repo or cfg.get('repo') or DEFAULT_REPO)
                branch=args.branch or cfg.get('branch') or 'main'
                if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_./-]*',branch) or '..' in branch: raise ValueError('Invalid branch')
                recorder.stage='fetch_revision'
                mirror=root/'repository.git'
                if not mirror.exists(): git('init','--bare',mirror)
                git('--git-dir',mirror,'fetch','--no-tags','--depth=1',repo,branch)
                commit=git('--git-dir',mirror,'rev-parse','FETCH_HEAD')
                attempted=root/'last_attempt.json'
                if args.if_new and ((root/'active.json').exists() and json.loads((root/'active.json').read_text())['release_id']==commit or
                    attempted.exists() and json.loads(attempted.read_text()).get('commit')==commit):
                    recorder.stage='no_new_revision'; recorder.finish(0); return
                atomic_json(attempted,{'commit':commit,'state':'checking'})
                if args.revision and args.revision!=commit: raise RuntimeError('Fetched HEAD differs from requested revision; no activation.')
                candidate=snapshot(mirror,commit,WORKSPACE)
                new={'release_id':commit,'commit':commit,'origin':repo}
                cfg.update(repo=repo,branch=branch)
            verify_source(candidate)
            syntax_check(candidate)
            if args.automatic:
                policy=json.loads((candidate/'DEPLOYMENT_POLICY.json').read_text())
                old_policy=json.loads((current/'DEPLOYMENT_POLICY.json').read_text()) if (current/'DEPLOYMENT_POLICY.json').exists() else {}
                changed=training_hash(current)!=training_hash(candidate) or recipe_hash(current)!=recipe_hash(candidate)
                if policy.get('training_change')=='new_campaign' and (changed or old_policy.get('training_change')!='new_campaign'):
                    args.new_campaign='phoenix14t_cslr_suite_'+new['release_id'][:12]
            report=compatibility(current,candidate,WORKSPACE,args.new_campaign)
            atomic_json(root/'update-report.json',report)
            print(json.dumps(report,indent=2),flush=True)
            recorder.stage='validate_candidate_docker'
            recorder.call(['docker','build','--build-arg','BASE_IMAGE='+args.base_image,
                           '--build-arg','RUN_TESTS=0',
                           '-t',image_name(candidate,args.base_image),'.'],cwd=candidate)
            if args.new_campaign: cfg['campaign']=args.new_campaign
            activate(WORKSPACE,new,old,cfg)
            (root/'monitor-build-failed.json').unlink(missing_ok=True)
            recorder.stage='activated'; recorder.finish(0)
            print('Update activated. Start with python run.py. Roll back with python update.py --rollback.')
        except BaseException as exc:
            recorder.finish(130 if isinstance(exc,KeyboardInterrupt) else 1,exc)
            raise


if __name__=='__main__':
    try: main()
    except KeyboardInterrupt: raise SystemExit(130)
    except (RuntimeError,ValueError,subprocess.CalledProcessError) as error:
        print(f'UPDATE STOPPED: {error}',file=sys.stderr); raise SystemExit(1)
