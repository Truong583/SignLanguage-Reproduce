import json
from pathlib import Path
import pytest
from scripts.report_wandb import archive_console,archive_saved_consoles


class Run:
    def __init__(self): self.saved=[]
    def save(self,path,**kwargs): self.saved.append((Path(path),kwargs))


def setup(tmp_path):
    root=tmp_path/'runs';log=root/'diagnostics/source-1/console.log';log.parent.mkdir(parents=True)
    spool=tmp_path/'spool';spool.mkdir()
    return root,log,spool,{'diagnostic_directory':'runs/diagnostics/source-1'}


def test_archive_replays_entire_log_and_resumes_at_byte_offset(tmp_path):
    root,log,spool,status=setup(tmp_path);run=Run();state={}
    text='Train tiếng Việt\n'+'epoch=1 micro_step=1/7092 loss=80\n'*10000
    log.write_text(text,encoding='utf-8',newline='')
    archive_console(run,root,spool,status,state)
    stream=state['console_archive']['source-1'];offset=stream['offset']
    assert offset==len(text.encode())
    files=[p for p,_ in run.saved if p.suffix=='.txt']
    assert ''.join(p.read_text(encoding='utf-8') for p in files)==text
    assert all(p.stat().st_size<512*1024 for p in files)
    archive_console(run,root,spool,status,state)
    assert len([p for p,_ in run.saved if p.suffix=='.txt'])==len(files)
    with log.open('a',encoding='utf-8',newline='') as handle: handle.write('epoch=1 micro_step=21/7092 loss=70\n')
    archive_console(run,root,spool,status,state)
    assert (spool/f'console_archive/source-1/{offset:012d}.txt').read_text().startswith('epoch=1 micro_step=21')


def test_archive_redacts_secrets_and_retains_partial_line_until_complete(tmp_path):
    root,log,spool,status=setup(tmp_path);run=Run();state={}
    log.write_text('wandb_v1_dummy_test_only\npartial',encoding='utf-8',newline='')
    archive_console(run,root,spool,status,state)
    text=(spool/'console_archive/source-1/000000000000.txt').read_text()
    assert 'wandb_v1_dummy_test_only' not in text and 'partial' not in text
    with log.open('a',encoding='utf-8') as handle: handle.write(' completed\n')
    archive_console(run,root,spool,status,state)
    files=state['console_archive']['source-1']['chunks']
    assert (spool/files[-1]).read_text()=='partial completed\n'


def test_archive_upload_queue_failure_does_not_advance_cursor(tmp_path):
    root,log,spool,status=setup(tmp_path);state={};log.write_text('retained\n')
    class Offline:
        def save(self,*args,**kwargs): raise ConnectionError('offline')
    with pytest.raises(ConnectionError): archive_console(Offline(),root,spool,status,state)
    assert state['console_archive']['source-1']['offset']==0
    archive_console(Run(),root,spool,status,state)
    assert state['console_archive']['source-1']['offset']==len(log.read_bytes())


def test_archive_does_not_read_paths_outside_diagnostics(tmp_path):
    root,log,spool,status=setup(tmp_path);log.write_text('safe\n');run=Run()
    assert not archive_console(run,root,spool,{'diagnostic_directory':'../private'},{})
    assert not run.saved


def test_new_observer_can_recover_existing_source_console_without_laptop(tmp_path):
    root,log,spool,status=setup(tmp_path);log.write_text('before laptop shutdown\nafter laptop shutdown\n')
    state={};archive_console(Run(),root,spool,status,state)
    new=Run();archive_console(new,root,spool,status,{})
    assert any(path.name=='000000000000.txt' for path,_ in new.saved)
    index=json.loads((spool/'console_archive/index.json').read_text())
    assert index['active']=='source-1'
    assert index['streams']['source-1']['chunks']==['console_archive/source-1/000000000000.txt']


def test_archive_detects_truncation_instead_of_silently_skipping_data(tmp_path):
    root,log,spool,status=setup(tmp_path);log.write_text('long original log\n');state={}
    archive_console(Run(),root,spool,status,state);log.write_text('new\n')
    with pytest.raises(RuntimeError,match='truncated'): archive_console(Run(),root,spool,status,state)


def test_backfill_older_host_logs_preserves_active_source_and_excludes_external_files(tmp_path):
    root,log,spool,status=setup(tmp_path);log.write_text('current\n')
    old=root/'diagnostics/older/console.log';old.parent.mkdir();old.write_text('older train\nepoch=1 micro_step=1/7092 loss=80\n')
    run=Run();state={};archive_console(run,root,spool,status,state)
    archive_saved_consoles(run,root,spool,status,state)
    index=json.loads((spool/'console_archive/index.json').read_text())
    assert index['active']=='source-1' and 'older' in index['streams']
    assert any(path.parent.name=='older' for path,_ in run.saved)
