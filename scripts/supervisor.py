"""Outbound-only supervisor. Updates happen between runs, never inside a running trainer."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import signal
import time
from scripts.deployment import OperationLock,active_release,atomic_json,state_dir,image_name,workspace_label
from scripts.diagnostics import redact,Recorder,read_tail


def identity(source):
    return hashlib.sha256((Path(source)/'BUNDLE_SHA256.json').read_bytes()).hexdigest()


def should_run(state,revision):
    return state.get('revision')!=revision or state.get('outcome') not in {'failed','completed'}


def monitor_command(workspace,source,image,name):
    uid=f'{os.getuid()}:{os.getgid()}' if hasattr(os,'getuid') else '1000:1000'
    command=['docker','run','--rm','--init','--name',name,'--label','signlanguage.monitor='+workspace_label(workspace),
      '--user',uid,'--read-only','--cap-drop','ALL','--security-opt','no-new-privileges',
      '--memory','1g','--memory-swap','1g','--cpus','1','--pids-limit','256',
      '--tmpfs','/tmp:rw,nosuid,nodev,size=256m','--network','bridge']
    mounts=[(source,'/source',True),(workspace/'runs','/observed',True),
      (workspace/'runs/telemetry','/spool',False),
      (workspace/'.updates/machine.json','/machine.json',True),
      (workspace/'.updates/wandb_key','/wandb_key',True)]
    for src,dst,ro in mounts: command+=['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
    return command+['--workdir','/source',image,'python','scripts/report_wandb.py']


def supervise(workspace,initial,args):
    workspace=Path(workspace); root=state_dir(workspace)
    if not (root/'machine.json').exists():
        raise RuntimeError('First run python setup_machine.py on this Linux machine, or use python run.py --once without remote monitoring.')
    cfg=json.loads((root/'machine.json').read_text())
    if hasattr(signal,'SIGTERM'):
        def stop(signum,frame): raise KeyboardInterrupt
        signal.signal(signal.SIGTERM,stop)
    base=cfg['base_image']; env=os.environ.copy(); env['SIGNLANGUAGE_WORKSPACE']=str(workspace)
    for folder in ('data','assets','runs','runs/telemetry'): (workspace/folder).mkdir(parents=True,exist_ok=True)
    status_path=workspace/'runs/supervisor.json'; state_path=root/'supervisor-state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {}
    observer=None; observer_revision=None; log=None; name='signlanguage-monitor-'+workspace_label(workspace)
    def status(phase,**extra):
        atomic_json(status_path,{'phase':phase,'heartbeat_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**state,**extra})
    def observer_start(source):
        nonlocal observer,log,observer_revision
        if observer is not None and observer.poll() is None:
            if observer_revision==identity(source): return
            subprocess.run(['docker','stop','--time','15',name],capture_output=True,check=False)
            observer.wait(timeout=30)
        if log: log.close()
        image=image_name(source,base)
        # Cached build, no test or training pass. Prevent simultaneous manual update.
        recorder=Recorder(workspace,source); recorder.stage='monitor_docker_build'
        try:
            with OperationLock(workspace):
                recorder.call(['docker','build','--build-arg','BASE_IMAGE='+base,'--build-arg','RUN_TESTS=0','-t',image,'.'],cwd=source)
            recorder.finish(0)
        except BaseException as exc:
            recorder.finish(130 if isinstance(exc,KeyboardInterrupt) else getattr(exc,'returncode',1),exc)
            raise
        log=(workspace/'runs/telemetry/monitor.log').open('a',encoding='utf-8')
        observer=subprocess.Popen(monitor_command(workspace,source,image,name),stdout=log,stderr=subprocess.STDOUT)
        observer_revision=identity(source)
    with OperationLock(workspace,'supervisor.lock'):
        if active_release(workspace) is None:
            import update
            with OperationLock(workspace):
                old=update.bootstrap_snapshot(workspace)
                update.activate(workspace,old,old,update.settings(workspace))
        try:
            while True:
                source=active_release(workspace); revision=identity(source)
                failed_path=root/'monitor-build-failed.json'
                failed=json.loads(failed_path.read_text()) if failed_path.exists() else {}
                blocked=failed.get('revision')==revision
                if not blocked:
                    try: observer_start(source)
                    except (OSError,subprocess.CalledProcessError) as exc:
                        atomic_json(failed_path,{'revision':revision,'error':redact(str(exc))})
                        print('Monitor/Docker setup failed. Diagnostics retained locally; waiting for a corrected GitHub revision. W&B cannot send logs until its image starts.',flush=True)
                        blocked=True
                if blocked:
                    deadline=time.monotonic()+max(30,int(cfg.get('poll_seconds',300)))
                    while time.monotonic()<deadline:
                        status('waiting_for_monitor_fix')
                        time.sleep(min(5,max(0,deadline-time.monotonic())))
                    subprocess.run([sys.executable,str(source/'update.py'),'--if-new','--automatic','--repo',cfg['repo'],
                      '--branch',cfg['branch'],'--base-image',base],env=env)
                    continue
                if should_run(state,revision):
                    state={'revision':revision,'outcome':'running'}; atomic_json(state_path,state); status('training')
                    process=subprocess.Popen([sys.executable,str(source/'run.py'),'--once','--base-image',base,*args],env=env,
                        start_new_session=os.name!='nt')
                    while process.poll() is None:
                        status('training',monitor_alive=observer.poll() is None)
                        time.sleep(5)
                    code=process.returncode
                    if code not in (0,130):
                        latest=workspace/'runs/status.json'
                        report=json.loads(latest.read_text()) if latest.exists() else {}
                        if report.get('status')!='failed':
                            crash=Recorder(workspace,source); crash.stage='worker_exit'
                            log_path=workspace/report.get('diagnostic_directory','runs/diagnostics/missing')/'console.log'
                            if log_path.resolve().is_relative_to((workspace/'runs/diagnostics').resolve()):
                                crash.line(read_tail(log_path))
                            crash.line(f'Worker exited with status {code}; review Docker/OS for external termination.\n')
                            crash.finish(code)
                    state.update(outcome='completed' if code==0 else 'interrupted' if code==130 else 'failed',exit_code=code)
                    atomic_json(state_path,state)
                    if code==130: return 130
                status('waiting_for_revision',monitor_alive=observer.poll() is None)
                # Every successful/failed revision runs at most once automatically.
                wait_until=time.monotonic()+max(30,int(cfg.get('poll_seconds',300)))
                while time.monotonic()<wait_until:
                    status('waiting_for_revision',monitor_alive=observer.poll() is None)
                    time.sleep(min(5,max(0,wait_until-time.monotonic())))
                status('checking_revision')
                result=subprocess.run([sys.executable,str(source/'update.py'),'--if-new','--automatic','--repo',cfg['repo'],
                  '--branch',cfg['branch'],'--base-image',base],env=env)
                if result.returncode: status('update_rejected_or_unavailable')
        except KeyboardInterrupt:
            status('stopped')
            if 'process' in locals() and process.poll() is None:
                if os.name!='nt': os.killpg(process.pid,signal.SIGINT)
                else: process.terminate()
                try: process.wait(timeout=120)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
            return 130
        finally:
            if observer and observer.poll() is None:
                subprocess.run(['docker','stop','--time','15',name],capture_output=True,check=False)
                try: observer.wait(timeout=30)
                except subprocess.TimeoutExpired: observer.kill(); observer.wait()
            if log: log.close()
