import io
import json
from pathlib import Path
import pytest
from scripts.privacy import require_private
from scripts.report_wandb import scalar_metrics,contained_file,scan,send_live_log
from scripts.supervisor import should_run,monitor_command
from scripts.diagnostics import redact


@pytest.mark.parametrize('access',['PRIVATE','TEAM','RESTRICTED'])
def test_telemetry_requires_existing_private_project(access):
    def opener(request,timeout):
        assert timeout==30 and request.full_url=='https://api.wandb.ai/graphql'
        return io.BytesIO(json.dumps({'data':{'project':{'name':'research','access':access}}}).encode())
    assert require_private('me','research','dummy-not-a-real-key',opener)['access']==access


@pytest.mark.parametrize('value',[{'data':{'project':None}}, {'data':{'project':{'access':'PUBLIC'}}},
    {'data':{'project':{'access':'OPEN'}}},{'errors':[{'message':'unsupported field'}]}])
def test_telemetry_fail_closed_for_public_missing_or_unknown_project(value):
    with pytest.raises(RuntimeError,match='Telemetry was not started'):
        require_private('me','research','dummy',lambda *a,**k:io.BytesIO(json.dumps(value).encode()))


def test_no_automatic_retry_of_failed_or_completed_revision():
    for outcome in ('failed','completed'):
        assert not should_run({'revision':'old','outcome':outcome},'old')
        assert should_run({'revision':'old','outcome':outcome},'new')
    assert should_run({'revision':'old','outcome':'running'},'old')
    assert should_run({'revision':'old','outcome':'interrupted'},'old')


def test_monitor_mounts_no_models_data_git_token_or_socket(tmp_path):
    source=tmp_path/'source'; command=monitor_command(tmp_path,source,'image','observer')
    mounts=[command[i+1] for i,v in enumerate(command) if v=='--mount']
    assert len(mounts)==5
    assert not any(v in ' '.join(mounts) for v in ('github_token','docker.sock','/data','/assets'))
    assert all('readonly' in v for v in mounts if 'dst=/spool' not in v)
    assert '--gpus' not in command and '--privileged' not in command
    assert 'signlanguage.workspace=' not in ' '.join(command)


def test_metrics_filter_excludes_strings_secrets_and_nonfinite_values():
    values=scalar_metrics({'epoch':2,'loss':0.4,'api_key':'secret','prediction':'private gloss',
       'dev':{'sequence':{'wer':12.3}},'seconds':float('nan')})
    assert values=={'epoch':2,'loss':0.4,'dev/sequence/wer':12.3}


def test_observer_reads_single_and_suite_histories_once_and_filters_sensitive_fields(tmp_path):
    root=tmp_path/'runs'; spool=tmp_path/'spool'; spool.mkdir()
    for name in ('phoenix14t_cslr/history.jsonl','campaign/main/history.jsonl'):
        path=root/name; path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps({'epoch':1,'loss':0.4,'dev':{'sequence':{'wer':20.0}},'api_key':'fake-sensitive-value'})+'\n')
    logged=[]
    run=type('Run',(),{'log':lambda self,value:logged.append(value)})()
    state={}; scan(run,root,spool,state)
    assert len(logged)==2
    assert any('phoenix14t_cslr/loss' in row for row in logged)
    assert any('campaign/main/dev/sequence/wer' in row for row in logged)
    assert 'fake-sensitive-value' not in json.dumps(logged)
    scan(run,root,spool,state); assert len(logged)==2


def test_attachments_must_be_contained_and_bounded(tmp_path):
    root=tmp_path/'runs'; root.mkdir(); path=root/'file'; path.write_text('small')
    assert contained_file(path,root,20)
    assert not contained_file(path,root,2)
    outside=tmp_path/'key'; outside.write_text('secret')
    assert not contained_file(outside,root,20)


def test_redaction_handles_git_http_header_and_prefix_tokens():
    value=redact('AUTHORIZATION: basic cHJpdmF0ZQ== github_pat_12345 ghp_12345 WANDB_API_KEY=key123')
    assert all(secret not in value for secret in ('cHJpdmF0ZQ','github_pat_12345','ghp_12345','key123'))
    assert 'wandb_v1_dummy_test_only' not in redact('wandb_v1_dummy_test_only')


@pytest.mark.parametrize('public',[True,False])
def test_setup_public_repo_skips_github_token_and_keeps_wandb_private(tmp_path,monkeypatch,public):
    import setup_machine as module
    import urllib.error
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module.sys,'argv',['setup_machine.py','--entity','example-team'])
    asked=[]; privacy=[]
    def prompt(label):
        asked.append(label)
        return 'dummy_github' if 'GitHub' in label else 'dummy_wandb'
    def opener(request,timeout):
        if not public and not request.get_header('Authorization'):
            raise urllib.error.HTTPError(request.full_url,404,'Private',{},None)
        return io.BytesIO(json.dumps({'private':not public}).encode())
    monkeypatch.setattr(module.getpass,'getpass',prompt)
    monkeypatch.setattr(module.urllib.request,'urlopen',opener)
    monkeypatch.setattr(module,'require_private',lambda *args:privacy.append(args))
    module.main()
    cfg=json.loads((tmp_path/'.updates/machine.json').read_text())
    assert cfg['public_repo']==public and privacy==[('example-team','signlanguage-reproduction','dummy_wandb')]
    assert len(asked)==(1 if public else 2)
    assert (tmp_path/'.updates/github_token').exists()==(not public)


def test_public_updates_send_no_github_auth_header(tmp_path,monkeypatch):
    import update as module
    monkeypatch.setattr(module,'WORKSPACE',tmp_path)
    root=tmp_path/'.updates'; root.mkdir()
    (root/'machine.json').write_text('{"public_repo":true}')
    (root/'github_token').write_text('dummy-stale-token')
    monkeypatch.delenv('GIT_CONFIG_COUNT',raising=False)
    assert 'GIT_CONFIG_VALUE_0' not in module.git_env()


def test_supervisor_waits_after_failure_and_updates_only_after_worker_exit(tmp_path,monkeypatch):
    import scripts.supervisor as module
    source=tmp_path/'source'; source.mkdir(); (source/'BUNDLE_SHA256.json').write_text('{}')
    root=tmp_path/'.updates'; root.mkdir()
    (root/'machine.json').write_text(json.dumps({'base_image':'base','repo':'repo','branch':'main','poll_seconds':30}))
    monkeypatch.setattr(module,'active_release',lambda w:source)
    clock=[0]; updates=[]; workers=[]
    class Process:
        def __init__(self,worker): self.worker=worker; self.returncode=1 if worker else None
        def poll(self): return self.returncode
        def wait(self,timeout=None): self.returncode=0; return 0
    def popen(command,**kwargs):
        worker=command[0]!='docker'; result=Process(worker)
        if worker: workers.append(result)
        return result
    def run(command,**kwargs):
        if '--if-new' in command:
            assert all(p.poll() is not None for p in workers)
            updates.append(command)
        return type('Result',(),{'returncode':1 if command[:3]==['docker','image','inspect'] else 0})()
    def sleep(seconds):
        clock[0]+=seconds
        if len(updates)>=2: raise KeyboardInterrupt
    monkeypatch.setattr(module.subprocess,'Popen',popen); monkeypatch.setattr(module.subprocess,'run',run)
    monkeypatch.setattr(module.Recorder,'call',lambda self,*a,**k:0)
    monkeypatch.setattr(module.time,'sleep',sleep); monkeypatch.setattr(module.time,'monotonic',lambda:clock[0])
    # Test changes no process-global signal handler.
    monkeypatch.setattr(module.signal,'signal',lambda *a:None)
    assert module.supervise(tmp_path,source,[])==130
    assert len(workers)==1 and len(updates)==2
    assert json.loads((root/'supervisor-state.json').read_text())['outcome']=='failed'
    reports=list((tmp_path/'runs/diagnostics').glob('*/diagnostic.json'))
    assert any(json.loads(p.read_text())['stage']=='worker_exit' for p in reports)


@pytest.mark.parametrize('recover',[False,True])
def test_failed_monitor_build_waits_for_updates_and_never_rebuilds_same_revision(tmp_path,monkeypatch,recover):
    import subprocess
    import scripts.supervisor as module
    old=tmp_path/'old'; old.mkdir(); (old/'BUNDLE_SHA256.json').write_text('{}')
    new=tmp_path/'new'; new.mkdir(); (new/'BUNDLE_SHA256.json').write_text('{"fixed":true}')
    root=tmp_path/'.updates'; root.mkdir()
    (root/'machine.json').write_text(json.dumps({'base_image':'base','repo':'repo','branch':'main','poll_seconds':30}))
    active=[old]; clock=[0]; builds=[]; workers=[]; updates=[]
    monkeypatch.setattr(module,'active_release',lambda w:active[0])
    def build(self,command,**kwargs):
        builds.append(kwargs['cwd'])
        if kwargs['cwd']==old:
            self.line('ninja 1.11.1.1 is not supported on this platform\n')
            raise subprocess.CalledProcessError(1,command)
        return 0
    class Process:
        def __init__(self,worker): self.returncode=1 if worker else None
        def poll(self): return self.returncode
        def wait(self,timeout=None): self.returncode=0; return 0
    def popen(command,**kwargs):
        worker=command[0]!='docker'
        if worker: workers.append(command)
        return Process(worker)
    def run(command,**kwargs):
        if '--if-new' in command:
            updates.append(command)
            if recover: active[0]=new
        return type('Result',(),{'returncode':1 if command[:3]==['docker','image','inspect'] else 0})()
    def sleep(seconds):
        clock[0]+=seconds
        if len(updates)>=2: raise KeyboardInterrupt
    monkeypatch.setattr(module.Recorder,'call',build)
    monkeypatch.setattr(module.subprocess,'Popen',popen); monkeypatch.setattr(module.subprocess,'run',run)
    monkeypatch.setattr(module.time,'sleep',sleep); monkeypatch.setattr(module.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(module.signal,'signal',lambda *a:None)
    assert module.supervise(tmp_path,old,[])==130
    assert builds==([old,new] if recover else [old])
    assert len(workers)==(1 if recover else 0) and len(updates)==2
    reports=[json.loads(p.read_text()) for p in (tmp_path/'runs/diagnostics').glob('*/diagnostic.json')]
    assert any(r['stage']=='monitor_docker_build' and r['exit_code']==1 for r in reports)
