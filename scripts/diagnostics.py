"""Classify execution failures and create bounded, redacted diagnostic bundles."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import threading
import time
import uuid
import zipfile


def redact(text):
    text=re.sub(r'-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----','[REDACTED PRIVATE KEY]',text,flags=re.S)
    text=re.sub(r'(?i)(Bearer|Basic)\s+[A-Za-z0-9_.~+/=-]+',r'\1 [REDACTED]',text)
    text=re.sub(r'\b(?:github_pat_|gh[pousr]_)[A-Za-z0-9_]+','[REDACTED GITHUB TOKEN]',text)
    text=re.sub(r'\bwandb_v1_[A-Za-z0-9_-]+','[REDACTED WANDB TOKEN]',text)
    text=re.sub(r'(?i)((?:refresh_token|access_token|client_secret|password|api_key)["\x27]?\s*[:=]\s*["\x27]?)[^"\x27\s,}&]+',r'\1[REDACTED]',text)
    text=re.sub(r'https://[^/\s@]+@','https://[REDACTED]@',text)
    text=re.sub(r'\b(?:1//|ya29\.)[A-Za-z0-9_.-]+','[REDACTED GOOGLE TOKEN]',text)
    return text


def classify(text,exit_code):
    if exit_code==130: return {'category':'interrupted','confidence':'high','next_step':'Rerun the same command to restore the saved checkpoint.'}
    patterns=[
      ('download_connection',r'TimeoutError: The read operation timed out|Download interrupted|URLError.*timed out','Retain the archive .part and resume after network/server recovery; this is not a training failure.'),
      ('package_platform',r'is not supported on this platform','Reinstall a compatible wheel inside the Docker image; retain pip check.'),
      ('cuda_oom',r'CUDA out of memory|OutOfMemoryError','Use more VRAM; do not silently change the paper recipe.'),
      ('nonfinite_loss',r'Nonfinite loss|(?:loss\s*[:=]\s*)(?:nan|[+-]?inf)\b','Inspect loss, precision and data; review whether prior training is valid.'),
      ('ctc_alignment',r'CTC alignment impossible|Impossible CTC alignment','Inspect the named sample, labels, frame lengths and augmentation.'),
      ('disk_space',r'No space left on device|Not enough scratch space|Less than 10 GiB|Not enough .*space','Free space outside retained checkpoints or use a larger workspace.'),
      ('drive_auth',r'invalid_grant|RefreshError|access_denied|HttpError[^\n]*401','Refresh the private OAuth credential on your trusted machine.'),
      ('checkpoint_storage',r'Checkpoint persistence failed|Persistent checkpoints exist, but none|checksum mismatch','Inspect storage/network and restore a committed generation.'),
      ('gpu_setup',r'cannot access CUDA|requires CUDA|no NVIDIA driver|not compiled with CUDA|no kernel image','Inspect GPU access and the pinned Docker/CUDA environment.'),
      ('checkpoint_incompatible',r'Checkpoint/config mismatch|Resume config changed|different code/config/data|checkpoint identity differs|Manifest hashes differ','Use matching code/config/data or start a separate campaign.'),
      ('data_missing',r'FileNotFoundError|No images|No frames|Missing frames directory','Inspect dataset extraction and the exact missing path.'),
    ]
    for category,pattern,next_step in patterns:
        if re.search(pattern,text,re.I):
            return {'category':category,'confidence':'log signature; needs review','next_step':next_step}
    if exit_code in (137,-9):
        return {'category':'process_killed','confidence':'possible RAM limit or external SIGKILL','next_step':'Inspect Docker/OS memory limits and the retained checkpoint.'}
    return {'category':'success' if exit_code==0 else 'unknown_failure','confidence':'exit status only',
            'next_step':'Measured correctness still requires evaluation.' if exit_code==0 else 'Send this diagnostic bundle for review; do not auto-restart indefinitely.'}


class Recorder:
    def __init__(self,workspace,source,global_status=True):
        self.source=Path(source)
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        self.root=Path(workspace)/'runs/diagnostics'/f'{stamp}-{uuid.uuid4().hex[:8]}'
        self.root.mkdir(parents=True,exist_ok=True)
        self.path=self.root/'console.log'; self.stage='startup'; self.private_block=False
        self.workspace=Path(workspace); self.last_output=time.monotonic(); self.code=None; self.alert=None
        self.global_status=global_status
        self.stop_event=threading.Event()
        self.thread=threading.Thread(target=self.monitor,daemon=True)
        self.thread.start()
    def line(self,line):
        if '-----BEGIN ' in line and 'PRIVATE KEY-----' in line:
            self.private_block=True; line='[REDACTED PRIVATE KEY]\n'
        elif self.private_block:
            if '-----END ' in line and 'PRIVATE KEY-----' in line: self.private_block=False
            return
        line=redact(line)
        self.last_output=time.monotonic()
        finding=classify(line,1)
        if finding['category']!='unknown_failure': self.alert=finding
        with self.path.open('a',encoding='utf-8') as f: f.write(line)
        print(line,end='',flush=True)
    def status(self):
        from scripts.deployment import atomic_json
        quiet=time.monotonic()-self.last_output
        value={'schema':1,'pid':os.getpid(),'stage':self.stage,'heartbeat_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'status':'running' if self.code is None else ('completed' if self.code==0 else 'interrupted' if self.code==130 else 'failed'),
               'seconds_without_console_output':round(quiet),'quiet_warning':quiet>30*60,
               'suspected_error':self.alert,'diagnostic_directory':str(self.root.relative_to(self.workspace))}
        atomic_json(self.root/'status.json',value)
        if self.global_status: atomic_json(self.workspace/'runs/status.json',value)
    def monitor(self):
        while not self.stop_event.is_set():
            try: self.status()
            except OSError: pass  # failure to write monitor state is not proof training failed
            self.stop_event.wait(30)
    def call(self,command,**kwargs):
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            text=True,encoding='utf-8',errors='replace',bufsize=1,**kwargs)
        try:
            for line in process.stdout: self.line(line)
            code=process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
            raise
        if code: raise subprocess.CalledProcessError(code,command)
        return code
    def finish(self,code,error=None):
        if error: self.line(redact(str(error))+'\n')
        if getattr(error,'stderr',None): self.line(redact(str(error.stderr))+'\n')
        tail=read_tail(self.path)
        report={'schema':1,'stage':self.stage,'exit_code':code,'classification':classify(tail,code),
            'python':platform.python_version(),'platform':platform.platform(),
            'source_manifest_sha256':hashlib.sha256((self.source/'BUNDLE_SHA256.json').read_bytes()).hexdigest() if (self.source/'BUNDLE_SHA256.json').exists() else None,
            'note':'No dataset, checkpoint, environment variables or credential files are attached. Log signatures cannot prove scientific correctness.'}
        (self.root/'diagnostic.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        (self.root/'log_tail.txt').write_text(redact(tail),encoding='utf-8')
        target=self.root/'diagnostics.zip'
        with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
            for name in ('diagnostic.json','log_tail.txt'): z.write(self.root/name,name)
            if (self.source/'RELEASE.json').is_file(): z.write(self.source/'RELEASE.json','RELEASE.json')
        self.code=code; self.stop_event.set(); self.thread.join(timeout=5); self.status()
        print(f'DIAGNOSTIC: {report["classification"]["category"]}; {target}',flush=True)
        return target


def read_tail(path,limit=256*1024):
    path=Path(path)
    if not path.exists(): return ''
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-limit)); return f.read(limit).decode('utf-8',errors='replace')
