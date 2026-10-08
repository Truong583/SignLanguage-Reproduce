"""Analyze an existing training log without modifying training or uploading it."""
import argparse
from pathlib import Path
from scripts.diagnostics import Recorder,read_tail

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('log',type=Path); p.add_argument('--exit-code',type=int,default=1)
    a=p.parse_args()
    if not a.log.is_file(): p.error('Log file does not exist')
    root=Path(__file__).resolve().parent
    recorder=Recorder(root,root,global_status=False); recorder.stage='manual_log_analysis'
    for line in read_tail(a.log).splitlines(keepends=True): recorder.line(line)
    recorder.finish(a.exit_code)
