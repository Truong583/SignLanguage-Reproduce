import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile
import io
import zipfile
import pytest

from scripts.deployment import (OperationLock,atomic_json,active_release,settings,verify_source)
from scripts.diagnostics import Recorder,classify,redact

ROOT=Path(__file__).resolve().parents[1]


def updater():
    spec=importlib.util.spec_from_file_location('test_updater',ROOT/'update.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def fixture_source(path,text='version1'):
    path.mkdir(parents=True,exist_ok=True)
    values={'run.py':'# launcher','update.py':'# updater','Dockerfile':'# fixture',
            'scripts/run_local.py':'# fixture','repro/net.py':text}
    for name,value in values.items():
        dest=path/name; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_text(value)
    inventory={name:hashlib.sha256((path/name).read_bytes()).hexdigest() for name in values}
    (path/'BUNDLE_SHA256.json').write_text(json.dumps(inventory))


def test_update_lock_excludes_run_and_releases_after_exit(tmp_path):
    with OperationLock(tmp_path):
        with pytest.raises(RuntimeError,match='running or updating'):
            with OperationLock(tmp_path): pass
    with OperationLock(tmp_path): pass


def test_atomic_activation_and_rollback_preserve_data(tmp_path):
    module=updater(); root=tmp_path/'workspace'; root.mkdir()
    for ident,text in [('a'*40,'one'),('b'*40,'two')]: fixture_source(root/'.updates/releases'/ident,text)
    data=root/'data/sample'; data.parent.mkdir(); data.write_bytes(b'owned dataset')
    old={'release_id':'a'*40}; new={'release_id':'b'*40}
    module.activate(root,new,old,{'campaign':'new-run'})
    assert active_release(root).name=='b'*40 and settings(root)['campaign']=='new-run'
    module.activate(root,old,new,{'campaign':'old-run'})
    assert active_release(root).name=='a'*40 and data.read_bytes()==b'owned dataset'


def test_training_change_requires_explicit_new_campaign(tmp_path):
    module=updater(); old=tmp_path/'old'; new=tmp_path/'new'; fixture_source(old); fixture_source(new,'different model')
    checkpoint=tmp_path/'runs/campaign/last.pt'; checkpoint.parent.mkdir(parents=True); checkpoint.write_bytes(b'retained')
    with pytest.raises(RuntimeError,match='not activated'): module.compatibility(old,new,tmp_path)
    assert module.compatibility(old,new,tmp_path,'new-campaign')['training_code_changed']
    assert checkpoint.read_bytes()==b'retained'
    with pytest.raises(ValueError): module.compatibility(old,new,tmp_path,'../escape')


def test_real_git_snapshot_is_pinned_and_source_only(tmp_path):
    module=updater(); source=tmp_path/'repo'; fixture_source(source)
    subprocess.run(['git','init',str(source)],check=True,capture_output=True)
    subprocess.run(['git','-C',str(source),'add','.'],check=True,capture_output=True)
    subprocess.run(['git','-C',str(source),'-c','user.name=Test','-c','user.email=test@example.invalid','commit','-m','Fixture'],check=True,capture_output=True)
    commit=subprocess.run(['git','-C',str(source),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
    workspace=tmp_path/'workspace'; workspace.mkdir()
    result=module.snapshot(source/'.git',commit,workspace)
    verify_source(result); assert result.name==commit and not (result/'.git').exists()
    assert all((result/name).is_dir() and not any((result/name).iterdir()) for name in ('data','assets','runs'))
    assert module.snapshot(source/'.git',commit,workspace)==result


@pytest.mark.parametrize('name,type_', [('../escape',tarfile.REGTYPE),('link',tarfile.SYMTYPE)])
def test_snapshot_rejects_escape_and_symlinks(tmp_path,monkeypatch,name,type_):
    module=updater(); workspace=tmp_path/'workspace'; workspace.mkdir()
    def fake_git(*args):
        output=Path(args[args.index('--output')+1])
        with tarfile.open(output,'w') as z:
            item=tarfile.TarInfo(name); item.type=type_; item.linkname='/outside'
            z.addfile(item,io.BytesIO(b'') if type_==tarfile.REGTYPE else None)
    monkeypatch.setattr(module,'git',fake_git)
    with pytest.raises(ValueError): module.snapshot('ignored','a'*40,workspace)
    assert not (workspace/'.updates/releases'/('a'*40)).exists()


@pytest.mark.parametrize('text,code,expected',[
  ('CUDA out of memory',1,'cuda_oom'),('Nonfinite loss: stopping',1,'nonfinite_loss'),
  ('FileNotFoundError: frames',1,'data_missing'),('No space left on device',1,'disk_space'),
  ('not a known error',137,'process_killed'),('done',0,'success'),('stopped',130,'interrupted')])
def test_error_classification(text,code,expected): assert classify(text,code)['category']==expected


def test_diagnostics_redacts_secrets_and_never_packages_checkpoint(tmp_path):
    source=tmp_path/'source'; fixture_source(source)
    checkpoint=tmp_path/'runs/last.pt'; checkpoint.parent.mkdir(); checkpoint.write_bytes(b'not for sharing')
    recorder=Recorder(tmp_path,source)
    recorder.line('refresh_token="private-refresh-value" Authorization: Bearer private-access-value\n')
    recorder.line('CUDA out of memory\n'); target=recorder.finish(1)
    with zipfile.ZipFile(target) as z:
        assert set(z.namelist())=={'diagnostic.json','log_tail.txt'}
        text=z.read('log_tail.txt').decode()
        assert 'private-refresh-value' not in text and 'private-access-value' not in text
        assert json.loads(z.read('diagnostic.json'))['classification']['category']=='cuda_oom'
    assert json.loads((tmp_path/'runs/status.json').read_text())['status']=='failed'


def test_manual_diagnostic_does_not_overwrite_live_monitor(tmp_path):
    source=tmp_path/'source'; fixture_source(source)
    status=tmp_path/'runs/status.json'; status.parent.mkdir(); status.write_text('{"status":"running"}')
    recorder=Recorder(tmp_path,source,global_status=False); recorder.finish(1)
    assert json.loads(status.read_text())['status']=='running'


def test_failed_activation_keeps_old_pointer(tmp_path,monkeypatch):
    module=updater(); root=tmp_path/'workspace'; root.mkdir()
    old={'release_id':'a'*40}; atomic_json(root/'.updates/active.json',old)
    actual=module.atomic_json
    def fail(path,value):
        if Path(path).name=='active.json': raise OSError('simulated disk failure')
        actual(path,value)
    monkeypatch.setattr(module,'atomic_json',fail)
    with pytest.raises(OSError): module.activate(root,{'release_id':'b'*40},old,{'campaign':'new'})
    assert json.loads((root/'.updates/active.json').read_text())==old
