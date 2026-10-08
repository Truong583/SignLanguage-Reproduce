import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader

from repro.data import SignDataset, collate
from repro.runtime import loader_options
from scripts.diagnostics import classify
from scripts.resource_diagnostics import memory_snapshot, exit_evidence


def test_cgroup_v2_distinguishes_kernel_oom_from_other_kill(tmp_path):
    group = tmp_path / 'container'
    group.mkdir()
    member = tmp_path / 'membership'
    member.write_text('0::/container\n')
    (group / 'memory.events').write_text('low 0\nhigh 0\nmax 3\noom 0\noom_kill 0\n')
    (group / 'memory.current').write_text('123')
    (group / 'memory.max').write_text(str(21 * 1024**3))
    (group / 'memory.peak').write_text('456')
    before = memory_snapshot(tmp_path, member)
    assert before['limit_bytes'] == 21 * 1024**3 and before['peak_bytes'] == 456
    unconfirmed = exit_evidence(-9, before, before)
    assert not unconfirmed['oom_kill_confirmed_in_cgroup']
    (group / 'memory.events').write_text('low 0\nhigh 0\nmax 7\noom 1\noom_kill 1\n')
    confirmed = exit_evidence(1, before, memory_snapshot(tmp_path, member))
    assert confirmed['oom_kill_confirmed_in_cgroup'] and confirmed['memory_event_delta']['oom_kill'] == 1
    # torchrun may exit 1 although its worker was SIGKILLed; inspect kernel data.
    assert classify('RESOURCE_EXIT: ' + json.dumps(confirmed), 1)['category'] == 'host_memory_oom'


def test_cgroup_v1_allocation_failures_are_not_oom_kill_proof(tmp_path):
    group = tmp_path / 'memory'
    group.mkdir()
    (group / 'memory.oom_control').write_text('oom_kill_disable 0\nunder_oom 0\noom_kill 2\n')
    (group / 'memory.failcnt').write_text('40')
    (group / 'memory.limit_in_bytes').write_text('21474836480')
    before = memory_snapshot(tmp_path, tmp_path / 'absent')
    assert before['version'] == 1 and before['allocation_failures'] == 40
    (group / 'memory.failcnt').write_text('43')
    after = memory_snapshot(tmp_path, tmp_path / 'absent')
    assert not exit_evidence(137, before, after)['oom_kill_confirmed_in_cgroup']


def test_memory_counters_absent_unlimited_or_reset_do_not_invent_oom(tmp_path):
    assert not memory_snapshot(tmp_path, tmp_path / 'absent')['available']
    (tmp_path / 'memory.events').write_text('oom 0\noom_kill 0\nmalformed\nmax nope\n')
    (tmp_path / 'memory.max').write_text('max')
    state = memory_snapshot(tmp_path, tmp_path / 'absent')
    assert state['limit_bytes'] is None and state['events'] == {'oom': 0, 'oom_kill': 0}
    assert not exit_evidence(-9, {'version': 2, 'events': {'oom_kill': 3}}, state)['oom_kill_confirmed_in_cgroup']


def test_torchrun_wrapped_sigkill_is_reported_without_claiming_oom():
    result = classify('exitcode  : -9 (pid: 69)\ntraceback : Signal 9 (SIGKILL) received by PID 69', 1)
    assert result['category'] == 'process_killed'
    assert 'distinguish' in result['next_step']


@pytest.mark.parametrize('mode', ['cpu', 'cpu_checkpoint', 'disk'])
def test_offload_stops_prefetch_and_pinned_full_clip_copies(mode):
    opts = loader_options({'workers': 8, 'activation_offload': mode}, True)
    assert opts == {'num_workers': 0, 'pin_memory': False, 'persistent_workers': False}
    assert 'prefetch_factor' not in opts  # DataLoader rejects prefetch_factor with zero workers.
    normal = loader_options({'workers': 2, 'low_memory_loader': False}, True)
    assert normal['num_workers'] == 2 and normal['prefetch_factor'] == 1


@pytest.mark.parametrize('workers', [-1, True, 1.5])
def test_invalid_worker_count_fails_before_dataset_loading(workers):
    with pytest.raises(ValueError, match='workers'):
        loader_options({'workers': workers}, False)


def test_low_memory_loader_keeps_real_rgb_samples_order_and_rng(tmp_path):
    rows = []
    for index in range(3):
        directory = tmp_path / f'clip{index}'
        directory.mkdir()
        rng = np.random.default_rng(index)
        for frame in range(4):
            pixels = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(directory / f'{frame:04}.png')
        rows.append({'id': str(index), 'frames': directory.name, 'gloss': 'A'})
    manifest = tmp_path / 'train.jsonl'
    manifest.write_text('\n'.join(map(json.dumps, rows)))
    dataset = SignDataset(manifest, tmp_path, train=True, seed=11)
    dataset.epoch = 2
    batches = []
    for cfg in ({'workers': 1}, {'workers': 1, 'low_memory_loader': False}):
        torch.manual_seed(22)
        expected_rng = torch.get_rng_state().clone()
        loader = DataLoader(dataset, batch_size=1, collate_fn=collate,
                            generator=torch.Generator().manual_seed(45), **loader_options(cfg, False))
        batches.append(list(loader))
        assert torch.equal(expected_rng, torch.get_rng_state())
    for left, right in zip(*batches):
        assert left['rows'] == right['rows']
        assert torch.equal(left['video'], right['video'])
        assert torch.equal(left['lengths'], right['lengths'])


def test_inplace_normalization_is_bitwise_equal_and_uses_one_clip_storage():
    original = torch.randint(0, 256, (9, 32, 32, 3), dtype=torch.uint8).permute(0, 3, 1, 2).float().div_(255)
    mean = original.new_tensor([.485, .456, .406])[None, :, None, None]
    std = original.new_tensor([.229, .224, .225])[None, :, None, None]
    expected = (original - mean) / std
    pointer = original.data_ptr()
    actual = original.sub_(mean).div_(std)
    assert actual.data_ptr() == pointer and torch.equal(actual, expected)
