import hashlib
import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch


def loader_options(cfg, use_cuda):
    """Bound host-side clip buffering when backward tensors also use host RAM.

    Dataset augmentation uses a per-sample RNG, so reducing worker concurrency
    does not change sampled frames, ordering, labels, or effective batch size.
    """
    requested = cfg.get('workers', 2)
    if not isinstance(requested, int) or isinstance(requested, bool) or requested < 0:
        raise ValueError('workers must be a nonnegative integer')
    memory_safe = cfg.get('low_memory_loader', True) and cfg.get('activation_offload', 'cpu_checkpoint') != 'none'
    workers = 0 if memory_safe else requested
    options = {'num_workers': workers, 'pin_memory': bool(use_cuda and not memory_safe),
               'persistent_workers': False}
    if workers:
        # One queued batch per worker; PyTorch's default would queue two.
        options['prefetch_factor'] = 1
    return options


def batch_plan(available, global_batch=6, micro_batch=1, requested=None):
    if global_batch<1 or micro_batch<1 or global_batch % micro_batch:
        raise ValueError('global_batch must be a positive multiple of micro_batch')
    if requested is not None:
        if requested<1 or requested>available or global_batch % (requested*micro_batch):
            raise ValueError('GPU count * micro_batch must divide global_batch exactly')
        world = requested
    else:
        choices = [n for n in range(1,available+1) if global_batch % (n*micro_batch)==0]
        if not choices:
            raise ValueError('No usable GPU/batch configuration')
        world = max(choices)
    return {'world_size':world,'micro_batch':micro_batch,'accumulation':global_batch//(world*micro_batch),
            'global_batch':global_batch}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state(state['cuda'])


def atomic_save(obj, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(obj,temporary)
    os.replace(temporary,path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def environment():
    return {'python':platform.python_version(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
            'platform':platform.platform(),'gpu':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}


def write_json(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
