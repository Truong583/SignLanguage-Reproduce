"""CPU-only observer: send approved scalars and sanitized diagnostics, never data/models."""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path,PurePosixPath
import re
import signal
import time
from scripts.deployment import atomic_json
from scripts.diagnostics import redact,read_tail
from scripts.privacy import require_private

ALLOWED={'epoch','loss','best','seconds','dropped_train_samples','wer','sub','del','ins'}


def scalar_metrics(record,prefix=''):
    out={}
    for key,value in record.items():
        if key=='dev' or key in {'sequence','frame','fused'}:
            if isinstance(value,dict): out.update(scalar_metrics(value,prefix+key+'/'))
        elif key in ALLOWED and isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value):
            out[prefix+key]=value
    return out


def contained_file(path,root,limit):
    path=Path(path); root=Path(root).resolve()
    return path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root) and path.stat().st_size<=limit


def live_console(root,directory):
    parts=PurePosixPath(str(directory).replace('\\','/')).parts
    if parts and parts[0]=='runs': parts=parts[1:]
    if len(parts)!=2 or parts[0]!='diagnostics' or parts[1] in {'.','..'}: return None
    root=Path(root); candidate=root.joinpath(*parts,'console.log')
    if candidate.is_file() and not candidate.is_symlink() and candidate.resolve().is_relative_to((root/'diagnostics').resolve()):
        return candidate
    return None


def send_live_log(run,root,spool,status):
    candidate=live_console(root,status.get('diagnostic_directory',''))
    if candidate is None: return False
    destination=Path(spool)/'live_log_tail.txt'
    temporary=destination.with_suffix('.txt.tmp')
    temporary.write_text(redact(read_tail(candidate,32*1024)),encoding='utf-8')
    os.replace(temporary,destination)
    run.save(str(destination),base_path=str(spool),policy='now')
    return True


def archive_console(run,root,spool,status,state,active_source=None):
    """Upload append-only sanitized console chunks independently of the laptop.

    Immutable files let a viewer recover logs after being offline. This reads
    only the active diagnostic console; dataset/model mounts are never scanned.
    Cursor persistence occurs after run.save has queued the chunk. New observer
    sessions restart the cursor and upload their own archive into their own run.
    """
    candidate=live_console(root,status.get('diagnostic_directory',''))
    if candidate is None: return False
    source=candidate.parent.name
    if not re.fullmatch(r'[A-Za-z0-9_-]+',source): return False
    streams=state.setdefault('console_archive',{})
    stream=streams.setdefault(source,{'offset':0,'chunks':[]})
    offset=int(stream['offset'])
    if candidate.stat().st_size<offset: raise RuntimeError('Console archive source was truncated')
    for _ in range(2):
        with candidate.open('rb') as handle:
            handle.seek(offset); content=handle.read(256*1024)
        boundary=content.rfind(b'\n')+1
        if not boundary: break
        content=content[:boundary]
        name=f'console_archive/{source}/{offset:012d}.txt'
        target=Path(spool)/name; target.parent.mkdir(parents=True,exist_ok=True)
        text=redact(content.decode('utf-8',errors='replace'))
        temporary=target.with_suffix('.tmp'); temporary.write_text(text,encoding='utf-8',newline=''); os.replace(temporary,target)
        run.save(str(target),base_path=str(spool),policy='now')
        stream['chunks'].append(name); offset+=len(content); stream['offset']=offset
    index=Path(spool)/'console_archive/index.json'
    atomic_json(index,{'version':1,'active':active_source or source,'streams':streams})
    run.save(str(index),base_path=str(spool),policy='now')
    return True


def archive_saved_consoles(run,root,spool,status,state):
    """Backfill one older host console per poll, without blocking training."""
    active=live_console(root,status.get('diagnostic_directory',''))
    if active is None: return
    for candidate in sorted((Path(root)/'diagnostics').glob('*/console.log'),reverse=True):
        if candidate==active or not contained_file(candidate,root,64*1024*1024): continue
        offset=state.get('console_archive',{}).get(candidate.parent.name,{}).get('offset',0)
        if candidate.stat().st_size<=offset: continue
        # A terminated source can end in an incomplete line. Complete log lines
        # are archived; do not let one such tail starve all older consoles.
        with candidate.open('rb') as handle:
            handle.seek(offset); peek=handle.read(256*1024)
        if b'\n' not in peek: continue
        archive_console(run,root,spool,{'diagnostic_directory':'diagnostics/'+candidate.parent.name},
                        state,active_source=active.parent.name)
        break


def scan(run,root,spool,state):
    # Exact names and containment checks; do not glob arbitrary files into artifacts.
    histories=set(root.glob('*/history.jsonl')) | set(root.glob('*/*/history.jsonl'))
    for path in sorted(histories):
        if not contained_file(path,root,32*1024*1024): continue
        name=path.relative_to(root).as_posix(); offset=state.setdefault('history',{}).get(name,0)
        if offset>path.stat().st_size: offset=0
        with path.open('rb') as stream:
            stream.seek(offset)
            for _ in range(200):
                line=stream.readline()
                if not line or not line.endswith(b'\n'): break
                try: values=scalar_metrics(json.loads(line))
                except (ValueError,TypeError): values={}
                if values: run.log({f'{name[:-14]}/{key}':value for key,value in values.items()})
                state['history'][name]=stream.tell()
    for report in sorted(root.glob('diagnostics/*/diagnostic.json')):
        if not contained_file(report,root,128*1024): continue
        name=report.parent.name
        if name in state.setdefault('sent',[]): continue
        value=json.loads(report.read_text())
        if value.get('exit_code') in (0,130,75): state['sent'].append(name); continue
        import wandb
        target=spool/'attachments'/name; target.mkdir(parents=True,exist_ok=True)
        artifact=wandb.Artifact('diagnostic-'+name,type='runtime-error')
        for filename in ('diagnostic.json','log_tail.txt'):
            source=report.parent/filename
            if contained_file(source,root,512*1024):
                destination=target/filename; destination.write_text(redact(source.read_text(errors='replace')),encoding='utf-8')
                artifact.add_file(str(destination),name=filename)
        run.log_artifact(artifact).wait()
        run.summary['last_error']=value.get('classification',{}).get('category','unknown_failure')
        run.summary['last_error_artifact']='diagnostic-'+name
        run.alert(title='SignLanguage stopped',text='Training/update stopped. Read runtime-error artifact diagnostic-'+name,
                  level='ERROR',wait_duration=300)
        state['sent'].append(name)


def main():
    root=Path('/observed'); spool=Path('/spool'); cfg=json.loads(Path('/machine.json').read_text())
    key=Path('/wandb_key').read_text().strip()
    # SDK never receives Git credentials, training source as code upload, or checkpoint mounts.
    os.environ.update(WANDB_API_KEY=key,WANDB_CONSOLE='off',WANDB_DISABLE_CODE='true',WANDB_DISABLE_GIT='true',
       WANDB_DIR=str(spool),WANDB_CACHE_DIR=str(spool/'cache'),WANDB_DATA_DIR=str(spool/'staging'),
       WANDB_CONFIG_DIR='/tmp/wandb-config',WANDB_ERROR_REPORTING='false')
    import wandb
    def stop(*_): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    observe(root,spool,cfg,key,wandb)


def observe(root,spool,cfg,key,wandb):
    state_file=spool/'cursor.json'
    state=json.loads(state_file.read_text()) if state_file.exists() else {}
    # Separate sessions prevent resuming a previously marked crashed W&B step counter.
    run=None
    try:
      while True:
        try:
            # Recheck visibility before sending anything, including after reconnect.
            require_private(cfg['wandb_entity'],cfg['wandb_project'],key)
            if run is None:
                run=wandb.init(entity=cfg['wandb_entity'],project=cfg['wandb_project'],job_type='observer',
                    group='teacher-machine',name='supervisor-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S'),
                    dir=str(spool),settings=wandb.Settings(console='off',disable_code=True,disable_git=True,
                      x_disable_stats=True,init_timeout=45),config={'scope':'PHOENIX14T-CSLR reconstruction','ci':False})
                # Refill a new session from source history. The previous SDK may
                # have queued metrics offline immediately before a power loss.
                state['history']={}
                state['console_archive']={}
            for filename in ('supervisor.json','status.json'):
                path=root/filename
                if not contained_file(path,root,128*1024): continue
                value=json.loads(path.read_text())
                for field in ('phase','status','stage','revision','outcome','exit_code','quiet_warning','monitor_alive','heartbeat_utc'):
                    if field in value: run.summary[filename+'/'+field]=value[field]
                if filename=='status.json':
                    run.log({'seconds_without_console_output':value.get('seconds_without_console_output',0)})
                    send_live_log(run,root,spool,value)
                    archive_console(run,root,spool,value,state)
                    archive_saved_consoles(run,root,spool,value,state)
            scan(run,root,spool,state)
            atomic_json(state_file,state)
            atomic_json(spool/'health.json',{'status':'connected','heartbeat_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'url':run.url})
        except Exception as exc:
            message=redact(str(exc)).replace(key,'[REDACTED]')
            atomic_json(spool/'health.json',{'status':'unavailable','message':message})
            print('TELEMETRY unavailable: '+message,flush=True)
            # Source logs/metrics remain on disk and are scanned again on recovery.
        time.sleep(30)
    except KeyboardInterrupt:
        pass
    finally:
        # Docker stops the observer on a source switch. Explicitly finish the
        # SDK session, otherwise the old dashboard run can remain Running.
        if run is not None:
            try: run.finish(exit_code=0)
            except Exception as exc:
                print('TELEMETRY finish unavailable: '+redact(str(exc)).replace(key,'[REDACTED]'),flush=True)
        atomic_json(spool/'health.json',{'status':'stopped','url':run.url if run is not None else None})


if __name__=='__main__': main()
