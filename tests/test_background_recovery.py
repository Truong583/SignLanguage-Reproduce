import json
from pathlib import Path
import pytest
from scripts.report_wandb import send_live_log,live_console
from scripts.deployment import ensure_image,image_name
from repro.persistence import LocalStore,publish
import service
import subprocess
import sys


def test_live_log_resolves_status_path_inside_runs_mount_and_redacts(tmp_path):
    root=tmp_path/'runs'; log=root/'diagnostics/sample/console.log'; log.parent.mkdir(parents=True)
    log.write_text('Dataset download: 1 GiB\nwandb_v1_dummy_test_only\n')
    spool=tmp_path/'spool'; spool.mkdir(); saved=[]
    run=type('Run',(),{'save':lambda self,path,**kwargs:saved.append((path,kwargs))})()
    assert send_live_log(run,root,spool,{'diagnostic_directory':'runs/diagnostics/sample'})
    assert live_console(root,'diagnostics/sample')==log
    assert 'Dataset download' in (spool/'live_log_tail.txt').read_text()
    assert 'wandb_v1_dummy_test_only' not in (spool/'live_log_tail.txt').read_text()
    assert saved[0][1]['policy']=='now'


@pytest.mark.parametrize('directory',['../private','/etc','runs/diagnostics/../../private','runs/runs/diagnostics/sample','diagnostics'])
def test_live_log_rejects_paths_outside_expected_diagnostic_layout(tmp_path,directory):
    assert live_console(tmp_path,directory) is None


def test_network_upload_failure_keeps_local_log(tmp_path):
    root=tmp_path/'runs'; log=root/'diagnostics/sample/console.log'; log.parent.mkdir(parents=True)
    log.write_text('progress retained'); spool=tmp_path/'spool'; spool.mkdir()
    class Run:
        def save(self,*args,**kwargs): raise ConnectionError('offline')
    with pytest.raises(ConnectionError): send_live_log(Run(),root,spool,{'diagnostic_directory':'runs/diagnostics/sample'})
    assert (spool/'live_log_tail.txt').read_text()=='progress retained' and log.read_text()=='progress retained'


def test_cached_image_needs_no_registry_or_build(tmp_path,monkeypatch):
    import scripts.deployment as module
    (tmp_path/'BUNDLE_SHA256.json').write_text('{}'); calls=[]
    def run(command,**kwargs):
        calls.append(command); return type('Result',(),{'returncode':0})()
    monkeypatch.setattr(module.subprocess,'run',run)
    assert ensure_image(tmp_path,'base')==image_name(tmp_path,'base')
    assert len(calls)==1 and calls[0][:3]==['docker','image','inspect']


def test_system_service_preserves_single_scope_uses_regular_user_and_orders_docker(tmp_path):
    text=service.unit(tmp_path/'project% name','/usr/bin/python3','annie')
    assert 'User=annie' in text and 'Group=docker' in text and 'User=root' not in text
    assert '_run --single' in text and '_run --suite' not in text
    assert 'Requires=docker.service' in text and 'RequiresMountsFor=' in text
    assert 'KillMode=mixed' in text and 'TimeoutStopSec=180' in text and '%% name' in text
    assert '_run --suite' in service.unit(tmp_path,'/usr/bin/python3','annie',True)
    assert service.service_name(tmp_path)!=service.service_name(tmp_path/'another')


def test_service_working_directory_is_a_scalar_path_not_a_quoted_argument():
    text=service.unit('/mnt/annie/Truong_K17/SignLanguage-Reproduce','/usr/bin/python3','annie')
    line=next(line for line in text.splitlines() if line.startswith('WorkingDirectory='))
    assert line=='WorkingDirectory=/mnt/annie/Truong_K17/SignLanguage-Reproduce'
    assert 'WorkingDirectory=/mnt/project%% name' in service.unit('/mnt/project% name','/usr/bin/python3','annie')
    assert 'ExecStart="/usr/bin/python3" "/mnt/project%% name/service.py"' in service.unit('/mnt/project% name','/usr/bin/python3','annie')


@pytest.mark.parametrize('path',['/mnt/invalid\npath','/mnt/invalid\rpath','/mnt/invalid\x00path'])
def test_service_rejects_control_characters_in_workspace(path):
    with pytest.raises(ValueError,match='Control characters'):
        service.unit(path,'/usr/bin/python3','annie')


def test_service_waits_for_docker_then_executes_single_run(monkeypatch):
    results=[type('R',(),{'returncode':1,'stdout':''})(),type('R',(),{'returncode':0,'stdout':'linux\n'})()]
    sleeps=[]; executed=[]
    monkeypatch.setattr(service.subprocess,'run',lambda *a,**k:results.pop(0))
    monkeypatch.setattr(service.time,'sleep',sleeps.append)
    def exec_(python,args): executed.append(args); raise RuntimeError('exec-observed')
    monkeypatch.setattr(service.os,'execv',exec_)
    with pytest.raises(RuntimeError,match='exec-observed'): service.daemon(False)
    assert sleeps==[5] and executed[0][-1]=='--single'


def test_single_runner_restores_previous_generation_after_corrupt_latest(tmp_path,monkeypatch):
    import scripts.run_local as module
    monkeypatch.setattr(module,'ROOT',tmp_path); monkeypatch.setattr(module,'code_hash',lambda:'same-numerical-code')
    monkeypatch.setattr(module,'training_finished',lambda *args:True)
    output=tmp_path/'runs/experiment'; output.mkdir(parents=True)
    store=LocalStore(output.parent/'.checkpoint-store/experiment')
    checkpoint=output/'last.pt'; checkpoint.write_bytes(b'older valid state'); publish(store,checkpoint)
    checkpoint.write_bytes(b'newer state'); publish(store,checkpoint)
    desc=json.loads((store.folder/store.commits('last')[0]).read_text())
    (store.folder/desc['payload']).write_bytes(b'incomplete after power failure')
    (output/'best.pt').write_bytes(b'best valid state')
    docs=tmp_path/'docs'; docs.mkdir(); (docs/'paper_targets.json').write_text('{}'); (docs/'AUDIT.md').write_text('limits')
    # Legacy saved config has no storage/cadence fields. Numerical fields remain strict.
    import yaml
    cfg={'output':str(output),'task':'cslr','epochs':50,'global_batch':6,'code_sha256':'same-numerical-code'}
    (output/'local_config.yaml').write_text(yaml.safe_dump(cfg)); calls=[]
    monkeypatch.setattr(module,'call',lambda *args:calls.append(args))
    module.train(cfg,Path('experiment.yaml'))
    assert checkpoint.read_bytes()==b'older valid state'
    assert cfg['checkpoint_every_updates']==200 and cfg['global_batch']==6 and cfg['epochs']==50
    assert '--resume' in calls[1]
    cfg['lr']=0.02
    with pytest.raises(ValueError,match='different config'): module.train(cfg,Path('experiment.yaml'))


def test_partial_single_training_cannot_evaluate_or_mark_completed(tmp_path,monkeypatch):
    import scripts.run_local as module
    import torch
    monkeypatch.setattr(module,'ROOT',tmp_path); monkeypatch.setattr(module,'code_hash',lambda:'same')
    output=tmp_path/'runs/experiment'; output.mkdir(parents=True)
    torch.save({'epoch':0,'epoch_complete':False},output/'last.pt')
    (output/'best.pt').write_bytes(b'best from an earlier epoch')
    calls=[]; monkeypatch.setattr(module,'call',lambda *args:calls.append(args))
    assert module.train({'output':str(output),'task':'cslr','epochs':50},Path('experiment.yaml')) is False
    assert len(calls)==2 and not (output/'COMPLETED.json').exists()


def test_real_tiny_cpu_checkpoint_resume_after_latest_generation_corruption(tmp_path):
    import numpy as np
    import torch
    import yaml
    from repro.persistence import restore
    data=tmp_path/'data'; data.mkdir(); output=tmp_path/'run'
    cfg={'task':'cslr','device':'cpu','input_kind':'features','feature_dim':16,'hidden_size':32,
      'data_root':str(data),'output':str(output),'global_batch':2,'micro_batch':1,'workers':0,
      'eval_batch':1,'epochs':1,'precision':'fp32','seed':0,'distillation':1.,'ctc_beam':2,
      'max_session_steps':1,'checkpoint_every_updates':1,
      'persistence':{'backend':'local','root':str(tmp_path/'store'),'run_id':'experiment'}}
    for split in ('train','dev','test'):
        rows=[]
        for i in range(4 if split=='train' else 1):
            name=f'{split}_{i}'; np.save(data/(name+'.npy'),np.random.default_rng(i).normal(size=(12,16)).astype('float32'))
            rows.append({'id':name,'features':name+'.npy','gloss':'a b','text':''})
        path=data/(split+'.jsonl'); path.write_text(''.join(json.dumps(r)+'\n' for r in rows)); cfg[split]=str(path)
    config=tmp_path/'config.yaml'; config.write_text(yaml.safe_dump(cfg))
    project=Path(__file__).resolve().parents[1]
    def train(*args):
        result=subprocess.run([sys.executable,'-m','repro.train','--config',str(config),*args],cwd=project,capture_output=True,text=True,timeout=90)
        assert result.returncode==0,result.stdout+result.stderr
    train()
    state=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
    assert not state['epoch_complete'] and state['optimizer_steps_in_epoch']==1
    store=LocalStore(tmp_path/'store/experiment'); desc=json.loads((store.folder/store.commits('last')[0]).read_text())
    (store.folder/desc['payload']).write_bytes(b'corrupt newest payload')
    restore(store,output/'last.pt')
    state=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
    assert state['optimizer_steps_in_epoch']==0
    cfg['max_session_steps']=0; config.write_text(yaml.safe_dump(cfg))
    train('--resume',str(output/'last.pt'))
    state=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
    assert state['epoch_complete'] and state['epoch']==0 and (output/'best.pt').is_file()
