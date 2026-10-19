#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/annie/Truong_K17/SignLanguage-Reproduce
UNIT=signlanguage-slt-worker.service
cd "$ROOT"
test "$(id -u)" != 0 || { echo 'Run this as annie, not as root.'; exit 1; }
test "$(id -un)" = annie || { echo 'This installer is for annie on the teacher machine.'; exit 1; }
if systemctl is-active --quiet signlanguage-8afccb21070c.service; then
  echo 'The old CSLR service is active. Stop installation to avoid conflicting jobs.'
  exit 1
fi
if systemctl is-active --quiet "$UNIT"; then
  echo 'SLT worker already active. Nothing changed.'
  exit 0
fi
sudo -v
sudo docker info --format '{{.OSType}}'
getent group docker >/dev/null
test -x /usr/bin/python3
mkdir -p "$ROOT/.updates/slt-worker" "$ROOT/runs/slt-worker"
chmod 700 "$ROOT/.updates/slt-worker" "$ROOT/runs/slt-worker"
cat > "$ROOT/.updates/slt-worker/worker.py" <<'SLT_WORKER_PY'
"""User-scoped SLT job worker. Only executes pinned jobs from the user's repo.

No SSH listener. Polling never replaces the code of an active job. A failed or
interrupted job is recorded and will not be retried without a new job_id.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import time
import urllib.request

ROOT = Path('/mnt/annie/Truong_K17/SignLanguage-Reproduce')
STATE = ROOT / '.updates/slt-worker'
RUNS = ROOT / 'runs/slt-worker'
REPO = 'https://raw.githubusercontent.com/Truong583/SignLanguage-Reproduce/'
MANIFEST = REPO + 'diagnostics/slt-probe-20261019/automation/slt-job.json'
LABEL = hashlib.sha256(str(ROOT).encode()).hexdigest()[:16]
stopping = False


def log(message):
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), message, flush=True)


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temp, path)


def download(url, limit):
    with urllib.request.urlopen(url, timeout=30) as response:
        value = response.read(limit + 1)
    if len(value) > limit:
        raise ValueError('Download exceeds size limit')
    return value


def validate(manifest):
    if manifest.get('schema') != 1:
        raise ValueError('Unsupported manifest')
    action = manifest.get('action')
    if action in ('idle', 'stop'):
        return manifest
    if action != 'run':
        raise ValueError('Unsupported action')
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', manifest.get('job_id', '')):
        raise ValueError('Invalid job id')
    if not re.fullmatch(r'[0-9a-f]{40}', manifest.get('commit', '')):
        raise ValueError('Job must pin a full commit SHA')
    if not re.fullmatch(r'[0-9a-f]{64}', manifest.get('sha256', '')):
        raise ValueError('Job must pin a SHA256')
    path = PurePosixPath(manifest.get('path', ''))
    if path.is_absolute() or '..' in path.parts or '\\' in str(path) or path.parts[:2] != ('automation', 'jobs') or path.suffix != '.sh':
        raise ValueError('Job must be an automation/jobs/*.sh file')
    minutes = manifest.get('timeout_minutes', 1440)
    if type(minutes) is not int or not 1 <= minutes <= 10080:
        raise ValueError('Invalid job duration')
    return manifest


def stop_containers(job_id):
    # Jobs must use both supplied labels. Never stop unrelated containers.
    result = subprocess.run(['docker', 'ps', '-q', '--filter', 'label=signlanguage.slt_worker=' + LABEL,
                             '--filter', 'label=signlanguage.slt_job=' + job_id],
                            capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError('Cannot inspect owned containers')
    for ident in result.stdout.split():
        subprocess.run(['docker', 'stop', '--time', '60', ident], check=True, timeout=90)


def signal_stop(*unused):
    global stopping
    stopping = True


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, signal_stop)
    signal.signal(signal.SIGINT, signal_stop)
    ledger_path = STATE / 'completed.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    status_path = STATE / 'status.json'
    old = json.loads(status_path.read_text()) if status_path.exists() else {}
    if old.get('state') == 'running':
        stop_containers(old['job_id'])
        ledger[old['job_id']] = {'status': 'interrupted', 'time': time.time()}
        write(ledger_path, ledger)
    process, active, stream, started = None, None, None, 0
    last_poll = 0
    requested = {'schema': 1, 'action': 'idle'}
    log('SLT worker ready. Waiting for an approved pinned job; no training started.')
    try:
        while not stopping:
            if process is not None and process.poll() is not None:
                # A job launcher must stay foreground until its training ends.
                stop_containers(active['job_id'])
                code = process.returncode
                ledger[active['job_id']] = {'exit_code': code, 'time': time.time()}
                write(ledger_path, ledger)
                write(status_path, {'state': 'job_completed' if code == 0 else 'job_failed',
                                    'job_id': active['job_id'], 'exit_code': code})
                log('Job ended: ' + active['job_id'] + ', exit=' + str(code))
                stream.close()
                process, active, stream = None, None, None
            if time.monotonic() - last_poll >= 120:
                last_poll = time.monotonic()
                try:
                    requested = validate(json.loads(download(MANIFEST, 65536)))
                    log('GitHub connected; action=' + requested['action'])
                except Exception:
                    requested = {'schema': 1, 'action': 'idle'}
                    log('Cannot read a valid GitHub instruction; retry in 120 seconds.')
            if process is not None:
                expired = time.monotonic() - started > active.get('timeout_minutes', 1440) * 60
                if requested['action'] == 'stop' or expired:
                    log('Stopping owned job at requested stop/time limit.')
                    os.killpg(process.pid, signal.SIGTERM)
                    stop_containers(active['job_id'])
                    try:
                        process.wait(timeout=65)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
            elif requested['action'] == 'run' and requested['job_id'] not in ledger:
                # The job code is immutable and hash checked before execution.
                job = dict(requested)
                source = download(REPO + job['commit'] + '/' + job['path'], 1024 * 1024)
                if hashlib.sha256(source).hexdigest() != job['sha256']:
                    raise ValueError('Job digest mismatch; refusing execution')
                path = STATE / (job['job_id'] + '.sh')
                path.write_bytes(source)
                logdir = RUNS / job['job_id']
                logdir.mkdir(exist_ok=True)
                stream = (logdir / 'worker-job.log').open('a', buffering=1)
                environment = dict(os.environ, SLT_WORKSPACE=str(ROOT), SLT_JOB_ID=job['job_id'],
                                   SLT_WORKER_LABEL=LABEL, SLT_NONINTERACTIVE='1')
                write(status_path, {'state': 'running', 'job_id': job['job_id'], 'commit': job['commit']})
                process = subprocess.Popen(['/bin/bash', str(path)], cwd=ROOT, env=environment,
                                           stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                active, started = job, time.monotonic()
                log('Started job: ' + job['job_id'])
            for _ in range(10):
                if stopping:
                    break
                time.sleep(1)
    finally:
        if process is not None:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            stop_containers(active['job_id'])
            try:
                process.wait(timeout=65)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            ledger[active['job_id']] = {'status': 'interrupted', 'time': time.time()}
            write(ledger_path, ledger)
            write(status_path, {'state': 'stopped', 'job_id': active['job_id']})
            if stream is not None:
                stream.close()


if __name__ == '__main__':
    main()

SLT_WORKER_PY
/usr/bin/python3 -c 'import ast,sys; ast.parse(open(sys.argv[1]).read())' "$ROOT/.updates/slt-worker/worker.py"
unit_file=$(mktemp)
cat > "$unit_file" <<EOF
[Unit]
Description=SignLanguage SLT update worker (waits for approved jobs)
After=network-online.target docker.service
Wants=network-online.target
Requires=docker.service

[Service]
Type=simple
User=annie
SupplementaryGroups=docker
WorkingDirectory=$ROOT
ExecStart=/usr/bin/python3 $ROOT/.updates/slt-worker/worker.py
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONDONTWRITEBYTECODE=1
Environment=PATH=/usr/local/bin:/usr/bin:/bin
Restart=on-failure
RestartSec=30
TimeoutStopSec=180
UMask=0077
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF
sudo install -m 644 "$unit_file" "/etc/systemd/system/$UNIT"
rm -f -- "$unit_file"
sudo systemctl daemon-reload
sudo systemctl enable --now "$UNIT"
sleep 3
systemctl is-active "$UNIT"
sudo journalctl -u "$UNIT" -n 8 --no-pager -o cat
printf '\nInstalled: worker checks GitHub every 120 seconds. Current job instruction is idle.\n'
printf 'Stop/disable later: sudo systemctl disable --now %s\n' "$UNIT"
