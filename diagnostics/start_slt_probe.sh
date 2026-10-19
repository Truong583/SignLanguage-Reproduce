#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/annie/Truong_K17/SignLanguage-Reproduce
SOURCE="$ROOT/.updates/releases/9c2792dafbb866649a9036433ed851b1e113b1a3"
CHECKPOINT="$ROOT/runs/phoenix14t_cslr_suite_9c2792dafbb8/phoenix14t_cslr"
IMAGE=mixsigngraph-repro:7d49da5a013e9554
GPU_NAME=signlanguage-slt-probe
WATCH_NAME=signlanguage-slt-probe-watch
cd "$ROOT"
sudo -v
sudo docker image inspect "$IMAGE" >/dev/null
for file in "$SOURCE/repro/model.py" "$CHECKPOINT/best.pt" "$ROOT/.updates/wandb_key" "$ROOT/.updates/machine.json" "$ROOT/assets/mbart-large-cc25/config.json"; do
  test -f "$file" || { printf 'Missing: %s\n' "$file"; exit 1; }
done
if systemctl is-active --quiet signlanguage-8afccb21070c.service; then
  echo 'Old training service is running. Stop here to avoid two training jobs.'
  exit 1
fi
for name in "$GPU_NAME" "$WATCH_NAME"; do
  if sudo docker container inspect "$name" >/dev/null 2>&1; then
    echo "Container $name already exists. This script does not start a duplicate."
    exit 1
  fi
done
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if test -n "$BUSY"; then
  echo 'A GPU compute process is already running. Do not start another job yet.'
  exit 1
fi
RUN_ID="slt-probe-$(date -u +%Y%m%dT%H%M%SZ)"
JOB=$(mktemp -d "$ROOT/runs/${RUN_ID}-XXXXXX")
mkdir -p "$JOB/code" "$JOB/report" "$JOB/spool"
chmod 700 "$JOB" "$JOB/code" "$JOB/report" "$JOB/spool"
cat > "$JOB/code/probe_slt_gpu.py" <<'SLT_PYTHON_END'
"""Disposable two-update SLT memory probe; never writes model checkpoints."""
import gc
import hashlib
import json
from pathlib import Path
import torch
from repro.data import SignDataset, collate, build_vocab, read_manifest
from repro.model import SignModel
from repro.runtime import seed_everything

torch.set_num_threads(2)
seed_everything(0)
if not torch.cuda.is_available():
    raise RuntimeError('CUDA unavailable')
torch.cuda.set_device(0)
torch.cuda.set_per_process_memory_fraction(0.85, 0)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
print('PyTorch allocator budget: 85% VRAM; CPU RAM limited by Docker.', flush=True)
checkpoint = torch.load('/checkpoint/best.pt', map_location='cpu', weights_only=False)
assert checkpoint['epoch'] + 1 == 22
cfg = dict(checkpoint['config'])
cfg.update(task='slt', mbart_path='/workspace/assets/mbart-large-cc25', language='de_DE',
           translation_checkpointing=True, translation_weight=1., label_smoothing=.2)
for split in ('train', 'dev', 'test'):
    assert hashlib.sha256(Path(cfg[split]).read_bytes()).hexdigest() == checkpoint['manifests'][split]
assert build_vocab(read_manifest(cfg['train']), cfg.get('target_field', 'gloss')) == checkpoint['vocab']
assert cfg.get('global_batch', 6) == 6
print('Loading CSLR + mBART...', flush=True)
model = SignModel(cfg, checkpoint['vocab'], initialize=False)
result = model.load_state_dict(checkpoint['model'], strict=False)
assert not result.unexpected_keys
assert all(k.startswith(('translation.', 'mapping.')) for k in result.missing_keys)
del checkpoint
gc.collect()
stage = 'move_model_to_gpu'
try:
    model = model.cuda().train()
    base, translation, mapping = [], [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (translation if name.startswith('translation.') else mapping if name.startswith('mapping.') else base).append(p)
    optimizer = torch.optim.Adam([
        {'params': base, 'lr': cfg.get('lr', 1e-4)},
        {'params': translation, 'lr': 1e-5},
        {'params': mapping, 'lr': 1e-3},
    ], weight_decay=cfg.get('weight_decay', 1e-4), foreach=False)
    # Avoid the extra parameter-sized temporary list of CUDA foreach Adam.
    # This is a disposable diagnostic, not a changed training checkpoint.
    scaler = torch.amp.GradScaler('cuda', init_scale=1.0)
    print('Probe settings: FP16 initial scale=1; Adam foreach=False.', flush=True)
    dataset = SignDataset(cfg['train'], cfg['data_root'], train=True, seed=cfg.get('seed', 0))
    assert len(dataset) >= 12
    for update in range(2):
        optimizer.zero_grad(set_to_none=True)
        for micro in range(6):
            stage = f'update_{update+1}_micro_{micro+1}'
            batch = collate([dataset[update * 6 + micro]])
            if not batch['rows'][0].get('text', '').strip():
                raise ValueError('Missing translation text')
            print(stage, batch['rows'][0]['id'], list(batch['video'].shape), flush=True)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = model(batch['video'].cuda(), batch['lengths'], batch['rows'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            scaler.scale(loss / 6).backward()
            del loss, batch
        stage = f'adam_update_{update+1}'
        print(stage, flush=True)
        scaler.unscale_(optimizer)
        bad = [name for name, p in model.named_parameters()
               if p.grad is not None and not torch.isfinite(p.grad).all().item()]
        if bad:
            print(json.dumps({'nonfinite_gradient_parameters': bad[:12],
                              'count': len(bad), 'amp_scale': scaler.get_scale()}), flush=True)
        if cfg.get('grad_clip', 0):
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        steps = [float(s['step']) for s in optimizer.state.values() if 'step' in s]
        if not steps or max(steps) != update + 1:
            raise FloatingPointError('AMP skipped an optimizer step; probe is inconclusive')
    print(json.dumps({'status': 'PASS_12_TRAIN_SAMPLES_ONLY',
                      'peak_allocated_GiB': torch.cuda.max_memory_allocated()/1024**3,
                      'peak_reserved_GiB': torch.cuda.max_memory_reserved()/1024**3,
                      'checkpoints_written': False}), flush=True)
except torch.OutOfMemoryError:
    print(json.dumps({'status': 'OOM_WITHIN_85_PERCENT_ALLOCATOR_BUDGET', 'stage': stage,
                      'peak_allocated_GiB': torch.cuda.max_memory_allocated()/1024**3,
                      'checkpoints_written': False}), flush=True)
    raise
SLT_PYTHON_END
cat > "$JOB/code/slt_probe_supervisor.py" <<'SLT_PYTHON_END'
"""Bound a disposable GPU probe and retain its log/exit status on the host."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

root = Path('/report')
started = time.time()
child = None


def save(value):
    temp = root / 'result.json.tmp'
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temp, root / 'result.json')


def stop(*unused):
    if child is not None and child.poll() is None:
        child.terminate()


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
status = 'probe_failed'
code = 1
with (root / 'probe.log').open('w', encoding='utf-8', buffering=1) as log:
    try:
        child = subprocess.Popen([sys.executable, '-u', '/job/probe_slt_gpu.py'],
                                 stdout=log, stderr=subprocess.STDOUT)
        try:
            code = child.wait(timeout=1800)
            status = 'passed_12_samples_only' if code == 0 else 'probe_failed'
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            status, code = 'probe_time_limit', 124
        log.write('\nPROBE_EXIT: ' + str(code) + '\n')
    finally:
        save({'status': status, 'exit_code': code,
              'seconds': round(time.time() - started, 1),
              'checkpoints_written': False, 'long_training_started': False})
sys.exit(code)
SLT_PYTHON_END
cat > "$JOB/code/slt_probe_observer.py" <<'SLT_PYTHON_END'
"""CPU-only upload of bounded diagnostic logs; no dataset/checkpoint access."""
import json
import os
from pathlib import Path
import time

from scripts.diagnostics import redact
from scripts.privacy import require_private

report, spool = Path('/report'), Path('/spool')
cfg = json.loads(Path('/machine.json').read_text())
key = Path('/wandb_key').read_text().strip()
run_id = os.environ['PROBE_RUN_ID']
os.environ.update(WANDB_API_KEY=key, WANDB_CONSOLE='off', WANDB_DISABLE_CODE='true',
                  WANDB_DISABLE_GIT='true', WANDB_ERROR_REPORTING='false',
                  WANDB_DIR=str(spool), WANDB_CACHE_DIR=str(spool / 'cache'),
                  WANDB_DATA_DIR=str(spool / 'staging'), WANDB_CONFIG_DIR='/tmp/wandb')
import wandb

deadline = time.monotonic() + 7200
run = None
while time.monotonic() < deadline:
    try:
        require_private(cfg['wandb_entity'], cfg['wandb_project'], key)
        if run is None:
            run = wandb.init(entity=cfg['wandb_entity'], project=cfg['wandb_project'],
                             id=run_id, name=run_id, resume='allow', job_type='slt-memory-probe',
                             dir=str(spool), settings=wandb.Settings(init_timeout=60))
            (spool / 'dashboard.txt').write_text(run.url, encoding='utf-8')
            print('Dashboard: ' + run.url, flush=True)
        if (report / 'probe.log').exists():
            with (report / 'probe.log').open('rb') as stream:
                stream.seek(max(0, (report / 'probe.log').stat().st_size - 512 * 1024))
                content = stream.read(512 * 1024).decode('utf-8', errors='replace')
            destination = spool / 'probe.log'
            destination.write_text(redact(content), encoding='utf-8')
            run.save(str(destination), base_path=str(spool), policy='now')
        if (report / 'result.json').exists():
            result = json.loads((report / 'result.json').read_text())
            run.summary.update(result)
            destination = spool / 'result.json'
            destination.write_text(json.dumps(result, indent=2), encoding='utf-8')
            run.save(str(destination), base_path=str(spool), policy='now')
            run.finish(exit_code=0 if result.get('exit_code') == 0 else 1)
            print('PROBE RESULT UPLOADED', flush=True)
            break
        run.summary['status'] = 'waiting_for_probe_result'
        run.log({'probe/log_age_seconds': max(0, time.time() - (report / 'probe.log').stat().st_mtime)
                 if (report / 'probe.log').exists() else 0})
        time.sleep(20)
    except Exception:
        # Never print raw exceptions that may contain credentials/HTTP headers.
        print('Upload unavailable; local results retained. Retrying in 60 seconds.', flush=True)
        time.sleep(60)
else:
    if run is not None:
        run.summary['status'] = 'observer_timeout_check_local_result'
        run.finish(exit_code=1)
    print('Observer stopped after two hours; logs remain on the host.', flush=True)
SLT_PYTHON_END

COMMON=(--init --user "$(id -u):$(id -g)" --read-only --cap-drop ALL
  --security-opt no-new-privileges --pids-limit 512
  --log-opt max-size=5m --log-opt max-file=2
  --mount "type=bind,src=$JOB/code,dst=/job,readonly"
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1)
sudo docker run -d --name "$WATCH_NAME" "${COMMON[@]}" \
  --network bridge --memory 1g --memory-swap 1g --cpus 1 \
  --tmpfs /tmp:rw,nosuid,nodev,size=256m \
  --mount "type=bind,src=$JOB/report,dst=/report,readonly" \
  --mount "type=bind,src=$JOB/spool,dst=/spool" \
  --mount "type=bind,src=$ROOT/.updates/machine.json,dst=/machine.json,readonly" \
  --mount "type=bind,src=$ROOT/.updates/wandb_key,dst=/wandb_key,readonly" \
  -e PYTHONPATH=/opt/signlanguage -e PROBE_RUN_ID="$RUN_ID" \
  --workdir /spool "$IMAGE" python -u /job/slt_probe_observer.py
if ! sudo docker run -d --name "$GPU_NAME" "${COMMON[@]}" \
  --gpus device=0 --network none --memory 21g --memory-swap 21g --cpus 2 \
  --shm-size 1g --stop-timeout 45 --tmpfs /tmp:rw,nosuid,nodev,size=512m \
  --mount "type=bind,src=$SOURCE,dst=/workspace,readonly" \
  --mount "type=bind,src=$ROOT/assets,dst=/workspace/assets,readonly" \
  --mount "type=bind,src=$ROOT/data,dst=/workspace/data,readonly" \
  --mount "type=bind,src=$CHECKPOINT,dst=/checkpoint,readonly" \
  --mount "type=bind,src=$JOB/report,dst=/report" \
  -e PYTHONPATH=/workspace -e HF_HOME=/tmp/hf \
  --workdir /workspace "$IMAGE" python -u /job/slt_probe_supervisor.py; then
  printf '%s\n' '{"status":"container_start_failed","exit_code":1}' > "$JOB/report/result.json"
  exit 1
fi
if command -v systemd-inhibit >/dev/null; then
  sudo -n nohup systemd-inhibit --what=sleep --mode=block \
    --who=SignLanguage-SLT-probe --why='Finish bounded SLT probe and upload its result' \
    sh -c 'docker wait signlanguage-slt-probe; docker wait signlanguage-slt-probe-watch' \
    > "$JOB/inhibit.log" 2>&1 < /dev/null &
fi
printf '\nProbe launched in background. Local folder: %s\n' "$JOB"
printf 'W&B run name: %s\n' "$RUN_ID"
printf 'Check upload connection: sudo docker logs --tail 15 %s\n' "$WATCH_NAME"
printf 'Local probe log: tail -n 30 "%s/report/probe.log"\n' "$JOB"
printf 'Stop probe if needed: sudo docker stop %s\n' "$GPU_NAME"
printf 'This is a bounded diagnostic only. It does not save checkpoints or start long training.\n'
