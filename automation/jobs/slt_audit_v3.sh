#!/usr/bin/env bash
set -euo pipefail
ROOT=${SLT_WORKSPACE:?Run through the SLT worker}
JOB_ID=${SLT_JOB_ID:?Missing job ID}
LABEL=${SLT_WORKER_LABEL:?Missing worker label}
SOURCE="$ROOT/.updates/releases/9c2792dafbb866649a9036433ed851b1e113b1a3"
CHECKPOINT="$ROOT/runs/phoenix14t_cslr_suite_9c2792dafbb8/phoenix14t_cslr"
IMAGE=mixsigngraph-repro:7d49da5a013e9554
GPU_NAME="slta-${JOB_ID}"
WATCH_NAME="sltw-${JOB_ID}"
cd "$ROOT"
docker info --format '{{.OSType}}'
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
test -z "$BUSY" || { echo 'GPU busy; refusing overlapping job.'; exit 2; }
JOB=$(mktemp -d "$ROOT/runs/${JOB_ID}-XXXXXX")
mkdir -p "$JOB/code" "$JOB/report" "$JOB/spool"
chmod 700 "$JOB" "$JOB/code" "$JOB/report" "$JOB/spool"
cat > "$JOB/code/probe_slt_gpu.py" <<'SLT_AUDIT_PY'
"""Disposable two-update SLT memory probe; never writes model checkpoints."""
import subprocess, sys, resource
subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                '/job/test_cpu_state_adam.py', '/job/test_slt_recipe_audit.py'], check=True)
import gc
import hashlib
import json
from pathlib import Path
import torch
from cpu_state_adam import CPUStateAdam
from repro.data import SignDataset, collate, build_vocab, read_manifest
from audited_slt_model import AuditedSLTModel as SignModel
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
           translation_checkpointing=True, translation_weight=1., label_smoothing=.2, text_beam=6, text_max_length=60)
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
    optimizer = CPUStateAdam([
        {'params': base, 'lr': cfg.get('lr', 1e-4)},
        {'params': translation, 'lr': 1e-5},
        {'params': mapping, 'lr': 1e-3},
    ], weight_decay=cfg.get('weight_decay', 1e-4))
    scaler = torch.amp.GradScaler('cuda', init_scale=1.0)
    print('Probe settings: FP16 initial scale=1; Adam moments on CPU; bounded CPU chunks.', flush=True)
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
        assert all(t.device.type == 'cpu' for state in optimizer.state.values()
                   for t in state.values() if isinstance(t, torch.Tensor))
        steps = [float(s['step']) for s in optimizer.state.values() if 'step' in s]
        if not steps or max(steps) != update + 1:
            raise FloatingPointError('AMP skipped an optimizer step; probe is inconclusive')
    optimizer.zero_grad(set_to_none=True)
    from repro.data import frame_paths
    print('Finding the longest source training video for a memory check...', flush=True)
    longest = max(range(len(dataset)), key=lambda i: len(frame_paths(dataset.root, dataset.rows[i])))
    stage = 'longest_training_video_backward'
    batch = collate([dataset[longest]])
    print(stage, batch['rows'][0]['id'], list(batch['video'].shape), flush=True)
    with torch.autocast('cuda', dtype=torch.float16):
        loss = model(batch['video'].cuda(), batch['lengths'], batch['rows'])
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite long-video loss')
    scaler.scale(loss / 6).backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all().item() for p in model.parameters()):
        raise FloatingPointError('Nonfinite long-video gradient')
    del loss, batch
    optimizer.zero_grad(set_to_none=True)
    stage = 'dev_generation'
    model.eval()
    dev = SignDataset(cfg['dev'], cfg['data_root'], train=False)
    longest_dev = max(range(len(dev)), key=lambda i: len(frame_paths(dev.root, dev.rows[i])))
    batch = collate([dev[longest_dev]])
    print(stage, batch['rows'][0]['id'], list(batch['video'].shape), flush=True)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        features = model.features(batch['video'].cuda(), batch['lengths'])
        translated = model.generate(features)
    assert len(translated) == 1
    print(json.dumps({'generation_completed': True, 'decoded_characters': len(translated[0]),
                      'quality_evaluated': False}), flush=True)
    print(json.dumps({'cpu_process_peak_GiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                      'recipe': 'upstream-language-labels-KL-beam6; FULL unpruned CC25',
                      'exact_paper_reproduction': False}), flush=True)
    print(json.dumps({'status': 'PASS_SLT_RUNTIME_SANITY_ONLY',
                      'peak_allocated_GiB': torch.cuda.max_memory_allocated()/1024**3,
                      'peak_reserved_GiB': torch.cuda.max_memory_reserved()/1024**3,
                      'checkpoints_written': False}), flush=True)
except torch.OutOfMemoryError:
    print(json.dumps({'status': 'OOM_WITHIN_85_PERCENT_ALLOCATOR_BUDGET', 'stage': stage,
                      'peak_allocated_GiB': torch.cuda.max_memory_allocated()/1024**3,
                      'checkpoints_written': False}), flush=True)
    raise
SLT_AUDIT_PY
cat > "$JOB/code/slt_probe_supervisor.py" <<'SLT_AUDIT_PY'
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
            status = 'passed_slt_runtime_sanity_only' if code == 0 else 'probe_failed'
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
SLT_AUDIT_PY
cat > "$JOB/code/slt_probe_observer.py" <<'SLT_AUDIT_PY'
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

deadline = time.monotonic() + 2400
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
SLT_AUDIT_PY
cat > "$JOB/code/cpu_state_adam.py" <<'SLT_AUDIT_PY'
"""Diagnostic Adam offload: CPU moments and bounded CPU update chunks.

Uses public torch.optim.Adam for the update itself (coupled weight decay,
not AdamW). Model parameters and gradients remain on their original device.
CPU/GPU arithmetic need not be bitwise identical. Not yet validated on the
teacher's GPU or integrated into the production training/resume runner.
"""
import copy
import torch


class CPUStateAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, chunk_elements=262144):
        if not isinstance(chunk_elements, int) or chunk_elements <= 0:
            raise ValueError('chunk_elements must be a positive integer')
        self.chunk_elements = chunk_elements
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        for group in self.param_groups:
            self._validate_group(group)
            for param in group['params']:
                self._validate_param(param)

    @staticmethod
    def _validate_group(group):
        if group['lr'] < 0 or group['eps'] < 0 or group['weight_decay'] < 0:
            raise ValueError('Invalid Adam hyperparameters')
        if not all(0 <= beta < 1 for beta in group['betas']):
            raise ValueError('Invalid Adam betas')
        for flag in ('amsgrad', 'maximize', 'differentiable', 'capturable', 'fused'):
            if group.get(flag, False):
                raise ValueError('Unsupported Adam option: ' + flag)

    @staticmethod
    def _validate_param(param):
        if param.dtype not in (torch.float32, torch.float64) or not param.is_contiguous():
            raise ValueError('CPUStateAdam requires contiguous FP32/FP64 parameters')

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            self._validate_group(group)
            for param in group['params']:
                if param.grad is None:
                    continue
                self._validate_param(param)
                if param.grad.is_sparse or not param.grad.is_contiguous():
                    raise ValueError('CPUStateAdam requires dense contiguous gradients')
                state = self.state[param]
                if not state:
                    state['step'] = torch.tensor(0., device='cpu')
                    state['exp_avg'] = torch.zeros_like(param, device='cpu')
                    state['exp_avg_sq'] = torch.zeros_like(param, device='cpu')
                flat = param.view(-1)
                grads = param.grad.view(-1)
                mean = state['exp_avg'].view(-1)
                variance = state['exp_avg_sq'].view(-1)
                for start in range(0, param.numel(), self.chunk_elements):
                    end = min(start + self.chunk_elements, param.numel())
                    # copy=True is required even in CPU tests: weight decay must
                    # not mutate the original gradient, and writes occur below.
                    chunk = flat[start:end].detach().to('cpu', copy=True)
                    chunk.grad = grads[start:end].detach().to('cpu', copy=True)
                    update = torch.optim.Adam([chunk], lr=group['lr'], betas=group['betas'],
                                              eps=group['eps'], weight_decay=group['weight_decay'],
                                              foreach=False)
                    update.state[chunk] = {'step': state['step'].clone(),
                                           'exp_avg': mean[start:end],
                                           'exp_avg_sq': variance[start:end]}
                    update.step()
                    flat[start:end].copy_(chunk)
                    del update, chunk
                # Every chunk uses the same previous counter; increment once.
                state['step'].add_(1)
        return loss

    def load_state_dict(self, saved):
        # The standard Optimizer loader would move moments onto the GPU.
        if len(saved['param_groups']) != len(self.param_groups):
            raise ValueError('Optimizer parameter group count differs')
        groups, states = [], {}
        for source, target in zip(saved['param_groups'], self.param_groups):
            if len(source['params']) != len(target['params']):
                raise ValueError('Optimizer parameter count differs')
            group = copy.deepcopy({k: v for k, v in source.items() if k != 'params'})
            group['params'] = target['params']
            self._validate_group(group)
            groups.append(group)
            for ident, param in zip(source['params'], target['params']):
                value = saved['state'].get(ident)
                if not value:
                    continue
                if set(value) != {'step', 'exp_avg', 'exp_avg_sq'}:
                    raise ValueError('Unexpected Adam checkpoint state')
                if any(value[k].shape != param.shape for k in ('exp_avg', 'exp_avg_sq')):
                    raise ValueError('Optimizer state tensor shape differs')
                step = torch.as_tensor(value['step']).detach().to('cpu')
                if step.numel() != 1 or not torch.isfinite(step).all() or step.item() < 0 or step.item() != int(step.item()):
                    raise ValueError('Invalid Adam step counter')
                states[param] = {'step': step.clone(),
                                 'exp_avg': value['exp_avg'].detach().to(device='cpu', dtype=param.dtype).clone(),
                                 'exp_avg_sq': value['exp_avg_sq'].detach().to(device='cpu', dtype=param.dtype).clone()}
        self.state.clear()
        self.state.update(states)
        self.param_groups[:] = groups
SLT_AUDIT_PY
cat > "$JOB/code/audited_slt_model.py" <<'SLT_AUDIT_PY'
"""Candidate SLT recipe aligned to available upstream translation source.

Still uses full CC25 vocabulary: author-pruned model/map_ids.pkl are missing.
This is NOT an assertion of exact paper reproduction. CSLR weights untouched.
"""
import math
import torch
from torch.nn import functional as F
from repro.model import SignModel
from repro.data import encode_targets


def translation_targets(ids, pad_id, lang_id):
    """Start decoder with language ID; exclude language and padding from loss."""
    if ids.ndim != 2 or ids.shape[1] < 2:
        raise ValueError('Expected mBART token sequences')
    labels = ids.clone()
    labels[(labels == pad_id) | (labels == lang_id)] = -100
    decoder = torch.full_like(ids, pad_id)
    decoder[:, 0] = lang_id
    shifted = ids[:, :-1].clone()
    shifted[shifted == lang_id] = pad_id
    decoder[:, 1:] = shifted
    return labels, decoder


def upstream_smoothed_kl(logits, labels, pad_id, smoothing):
    """Upstream XentLoss distribution, without allocating a dense target matrix.

Gold has mass 1-eps; pad has zero; remaining V-2 entries have eps/(V-2).
Preserves the KL entropy constant and the sum reduction of upstream XentLoss.
"""
    if not 0 <= smoothing < 1:
        raise ValueError('Invalid smoothing')
    size = logits.shape[-1]
    if size < 3 or not 0 <= pad_id < size:
        raise ValueError('Invalid vocabulary')
    logp = F.log_softmax(logits.float(), dim=-1)
    valid = (labels != -100) & (labels != pad_id)
    safe = labels.masked_fill(~valid, 0)
    nll = -logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    if smoothing == 0:
        return nll.masked_select(valid).sum()
    confidence = 1 - smoothing
    other = smoothing / (size - 2)
    smooth = -logp.sum(-1) + logp[..., pad_id]
    entropy_term = confidence * math.log(confidence) + smoothing * math.log(other)
    values = (confidence - other) * nll + other * smooth + entropy_term
    return values.masked_select(valid).sum()


class AuditedSLTModel(SignModel):
    def forward(self, video, lengths, rows=None):
        out = self.features(video, lengths)
        if rows is None:
            return out
        labels, label_lengths, required = encode_targets(rows, self.vocab, self.cfg.get('target_field', 'gloss'))
        if (required > out['lengths']).any():
            raise ValueError('CTC alignment impossible; labels are not truncated')
        loss = out['sequence'].float().sum() * 0
        for head, weight in [('conv', self.cfg.get('conv_ctc', 1.)), ('sequence', self.cfg.get('seq_ctc', 1.))]:
            if weight:
                loss = loss + weight * F.ctc_loss(out[head].float().log_softmax(-1), labels,
                    out['lengths'], label_lengths, blank=0, reduction='none', zero_infinity=False).mean()
        if self.cfg.get('distillation', 25.):
            loss = loss + self.cfg.get('distillation', 25.) * self.kd(
                out['conv'].float(), out['sequence'].detach().float(), use_blank=False)
        if self.translation is not None:
            ids = self.tokenizer([row['text'] for row in rows], return_tensors='pt', padding=True).input_ids.to(video.device)
            labels, decoder = translation_targets(ids, self.tokenizer.pad_token_id,
                self.tokenizer.lang_code_to_id[self.cfg.get('language', 'de_DE')])
            translated = self.translation(**self.translation_inputs(out['features'], out['lengths']),
                                          decoder_input_ids=decoder)
            loss = loss + self.cfg.get('translation_weight', 1.) * upstream_smoothed_kl(
                translated.logits, labels, self.tokenizer.pad_token_id, self.cfg.get('label_smoothing', .2)) / len(rows)
        return loss

    @torch.no_grad()
    def generate(self, out):
        if self.translation is None:
            return [''] * len(out['lengths'])
        generated = self.translation.generate(**self.translation_inputs(out['features'], out['lengths']),
            decoder_start_token_id=self.tokenizer.lang_code_to_id[self.cfg.get('language', 'de_DE')],
            num_beams=self.cfg.get('text_beam', 6), max_length=self.cfg.get('text_max_length', 60),
            length_penalty=self.cfg.get('text_length_penalty', 1.), use_cache=True)
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)
SLT_AUDIT_PY
cat > "$JOB/code/test_cpu_state_adam.py" <<'SLT_AUDIT_PY'
"""Numerical checks; does not launch the research model or touch its files."""
import copy
import pytest
import torch
from cpu_state_adam import CPUStateAdam


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('chunk', [1, 7, 10000])
def test_matches_adam_and_round_trip(dtype, chunk):
    torch.manual_seed(391)
    left = [torch.nn.Parameter(torch.randn(3, 11, dtype=dtype)),
            torch.nn.Parameter(torch.randn(9, dtype=dtype))]
    right = [torch.nn.Parameter(p.detach().clone()) for p in left]
    make = lambda params: [{'params': [params[0]], 'lr': 1e-4},
                           {'params': [params[1]], 'lr': 1e-3, 'weight_decay': 0.03}]
    reference = torch.optim.Adam(make(left), weight_decay=1e-4, foreach=False)
    candidate = CPUStateAdam(make(right), weight_decay=1e-4, chunk_elements=chunk)
    # Different CPU vector widths can round differently at chunk boundaries.
    tolerance = dict(rtol=2e-6, atol=2e-8) if dtype == torch.float32 else dict(rtol=1e-12, atol=1e-14)
    for index in range(8):
        for a, b in zip(left, right):
            a.grad = torch.randn_like(a)
            b.grad = a.grad.clone()
        if index == 2:
            left[1].grad = right[1].grad = None
        original_grads = [p.grad.clone() if p.grad is not None else None for p in right]
        reference.step()
        candidate.step()
        for a, b, gradient in zip(left, right, original_grads):
            torch.testing.assert_close(a, b, **tolerance)
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(reference.state[a][key], candidate.state[b][key], **tolerance)
                assert candidate.state[b][key].device.type == 'cpu'
            if gradient is not None:
                torch.testing.assert_close(gradient, b.grad, rtol=0, atol=0)
        if index == 3:
            checkpoint = copy.deepcopy(candidate.state_dict())
            candidate = CPUStateAdam(make(right), chunk_elements=chunk)
            candidate.load_state_dict(checkpoint)


def test_rejects_unsupported_and_wrong_state_shape():
    with pytest.raises(ValueError):
        CPUStateAdam([torch.nn.Parameter(torch.ones(3, dtype=torch.float16))])
    with pytest.raises(ValueError):
        CPUStateAdam([{'params': [torch.nn.Parameter(torch.ones(3))], 'amsgrad': True}])
    p = torch.nn.Parameter(torch.ones(3))
    optimizer = CPUStateAdam([p])
    p.grad = torch.ones_like(p)
    optimizer.step()
    saved = copy.deepcopy(optimizer.state_dict())
    saved['state'][0]['exp_avg'] = torch.zeros(4)
    with pytest.raises(ValueError):
        optimizer.load_state_dict(saved)


def test_resume_exactly_matches_uninterrupted_offload():
    torch.manual_seed(72)
    p = torch.nn.Parameter(torch.randn(57))
    first = CPUStateAdam([p], chunk_elements=13)
    p.grad = torch.randn_like(p)
    first.step()
    q = torch.nn.Parameter(p.detach().clone())
    second = CPUStateAdam([q], chunk_elements=13)
    second.load_state_dict(copy.deepcopy(first.state_dict()))
    for _ in range(4):
        p.grad = torch.randn_like(p)
        q.grad = p.grad.clone()
        first.step()
        second.step()
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(first.state[p][key], second.state[q][key], rtol=0, atol=0)
SLT_AUDIT_PY
cat > "$JOB/code/test_slt_recipe_audit.py" <<'SLT_AUDIT_PY'
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from audited_slt_model import AuditedSLTModel, translation_targets, upstream_smoothed_kl


def test_language_shift_and_labels_match_upstream():
    ids = torch.tensor([[5, 6, 2, 4], [7, 2, 4, 1]])
    labels, decoder = translation_targets(ids, 1, 4)
    assert labels.tolist() == [[5, 6, 2, -100], [7, 2, -100, -100]]
    assert decoder.tolist() == [[4, 5, 6, 2], [4, 7, 2, 1]]
    assert ids.tolist() == [[5, 6, 2, 4], [7, 2, 4, 1]]


@pytest.mark.parametrize('smoothing', [0., .2])
def test_kl_matches_actual_upstream_class_value_and_gradient(smoothing):
    # Extract the exact class without importing upstream's missing assets.
    import repro.model
    source = Path(repro.model.__file__).resolve().parents[1] / 'MixSignGraph/modules/translation.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'XentLoss')
    scope = {'torch': torch, 'nn': torch.nn, 'Variable': torch.autograd.Variable}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), scope)
    torch.manual_seed(2)
    left = torch.randn(2, 4, 13, requires_grad=True)
    right = left.detach().clone().requires_grad_(True)
    labels = torch.tensor([[3, 5, 2, -100], [6, 2, -100, -100]])
    original_labels = labels.masked_fill(labels == -100, 1)
    expected = scope['XentLoss'](pad_index=1, smoothing=smoothing)(left.log_softmax(-1), original_labels)
    actual = upstream_smoothed_kl(right, labels, 1, smoothing)
    torch.testing.assert_close(expected, actual, rtol=2e-6, atol=2e-6)
    expected.backward()
    actual.backward()
    torch.testing.assert_close(left.grad, right.grad, rtol=2e-5, atol=2e-7)


def test_ce_routes_gradients_into_video_mapping_and_mbart(monkeypatch):
    torch.set_num_threads(2)
    from transformers import MBartConfig, MBartForConditionalGeneration
    from repro.data import collate
    class Tokenizer:
        pad_token_id = 1
        eos_token_id = 2
        lang_code_to_id = {'de_DE': 4}
        def __call__(self, texts, **kwargs):
            return SimpleNamespace(input_ids=torch.tensor([[5, 6, 2, 4] for text in texts]))
        def batch_decode(self, values, **kwargs):
            return ['decoded' for row in values]
    config = MBartConfig(vocab_size=16, d_model=16, encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32,
        max_position_embeddings=64, pad_token_id=1, bos_token_id=0, eos_token_id=2)
    monkeypatch.setattr('repro.model.load_translation', lambda *args: (Tokenizer(), MBartForConditionalGeneration(config)))
    cfg = dict(task='slt', input_kind='features', feature_dim=8, hidden_size=16,
               mbart_path='unused', translation_checkpointing=False, conv_ctc=0., seq_ctc=0., distillation=0.,
               text_beam=2, text_max_length=5)
    model = AuditedSLTModel(cfg, ['<blank>', '<unk>', 'a', 'b'])
    batch = collate([(torch.randn(12, 8), dict(id='test', gloss='a b', text='sentence'))])
    batch['video'].requires_grad_()
    loss = model(batch['video'], batch['lengths'], batch['rows'])
    loss.backward()
    for value in [batch['video'].grad, model.mapping[0].weight.grad,
                  model.translation.model.shared.weight.grad]:
        assert value is not None and torch.isfinite(value).all() and value.abs().sum() > 0
    # Classifier is not on the translation path: feeding decoded glosses would
    # break the CE gradient route into the continuous visual feature network.
    model.eval()
    with torch.no_grad():
        assert model.generate(model.features(batch['video'], batch['lengths'])) == ['decoded']
SLT_AUDIT_PY

COMMON=(--init --user "$(id -u):$(id -g)" --read-only --cap-drop ALL
  --security-opt no-new-privileges --pids-limit 512
  --label "signlanguage.slt_worker=$LABEL" --label "signlanguage.slt_job=$JOB_ID"
  --log-opt max-size=5m --log-opt max-file=2
  --mount "type=bind,src=$JOB/code,dst=/job,readonly"
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1)
cleanup() { docker stop --time 60 "$GPU_NAME" "$WATCH_NAME" >/dev/null 2>&1 || true; }
trap cleanup TERM INT EXIT
docker run -d --name "$WATCH_NAME" "${COMMON[@]}" \
  --network bridge --memory 1g --memory-swap 1g --cpus 1 \
  --tmpfs /tmp:rw,nosuid,nodev,size=256m \
  --mount "type=bind,src=$JOB/report,dst=/report,readonly" \
  --mount "type=bind,src=$JOB/spool,dst=/spool" \
  --mount "type=bind,src=$ROOT/.updates/machine.json,dst=/machine.json,readonly" \
  --mount "type=bind,src=$ROOT/.updates/wandb_key,dst=/wandb_key,readonly" \
  -e PYTHONPATH=/opt/signlanguage -e PROBE_RUN_ID="$JOB_ID" \
  --workdir /spool "$IMAGE" python -u /job/slt_probe_observer.py
if ! docker run -d --name "$GPU_NAME" "${COMMON[@]}" \
  --gpus device=0 --network none --memory 21g --memory-swap 21g --cpus 2 \
  --shm-size 1g --stop-timeout 45 --tmpfs /tmp:rw,nosuid,nodev,size=512m \
  --mount "type=bind,src=$SOURCE,dst=/workspace,readonly" \
  --mount "type=bind,src=$ROOT/assets,dst=/workspace/assets,readonly" \
  --mount "type=bind,src=$ROOT/data,dst=/workspace/data,readonly" \
  --mount "type=bind,src=$CHECKPOINT,dst=/checkpoint,readonly" \
  --mount "type=bind,src=$JOB/report,dst=/report" \
  -e PYTHONPATH=/job:/workspace -e HF_HOME=/tmp/hf \
  --workdir /workspace "$IMAGE" python -u /job/slt_probe_supervisor.py; then
  printf '%s\n' '{"status":"container_start_failed","exit_code":1}' > "$JOB/report/result.json"
  docker wait "$WATCH_NAME" || true
  exit 1
fi
code=$(docker wait "$GPU_NAME")
if ! test -f "$JOB/report/result.json"; then
  oom=$(docker inspect --format '{{.State.OOMKilled}}' "$GPU_NAME")
  printf '{"status":"container_exited_without_result","exit_code":%s,"docker_oom_killed":%s}\n' "$code" "$oom" > "$JOB/report/result.json.tmp"
  mv "$JOB/report/result.json.tmp" "$JOB/report/result.json"
fi
docker wait "$WATCH_NAME"
echo "Audit done; GPU exit=$code; report=$JOB/report"
exit "$code"
