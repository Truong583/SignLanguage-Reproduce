"""Recompute bounded model segments without advancing BatchNorm buffers twice.

An outer ``save_on_cpu`` context may offload checkpoint boundary inputs. This
helper deliberately adds no inner saved-tensor hook: PyTorch's non-reentrant
checkpoint owns the hooks while it discards and regenerates segment interiors.
"""
from contextlib import contextmanager, nullcontext

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def _batchnorm_modules(modules):
    if isinstance(modules, nn.Module):
        modules = (modules,)
    seen = set()
    result = []
    for root in modules:
        if not isinstance(root, nn.Module):
            raise TypeError('Checkpoint modules must be torch.nn.Module instances')
        for module in root.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm) and id(module) not in seen:
                seen.add(id(module))
                result.append(module)
    return result


@contextmanager
def _recomputation_buffers(modules):
    # Keep training=True and the same BatchNorm operation in both passes. Only
    # recomputation's mutable running buffers are private copies. In particular,
    # never copy_ back into a tensor saved by autograd: that changes its version.
    originals = []
    try:
        for module in modules:
            for name in ('running_mean', 'running_var', 'num_batches_tracked'):
                value = module._buffers.get(name)
                originals.append((module, name, value))
                if value is not None:
                    module._buffers[name] = value.detach().clone()
        yield
    finally:
        for module, name, value in reversed(originals):
            module._buffers[name] = value


def checkpoint_call(function, *args, modules=(), enabled=True, **kwargs):
    """Checkpoint one segment, preserving RNG and original BatchNorm objects.

    ``modules`` lists all modules invoked by a function closure. A directly
    passed module is discovered automatically. Do not include arbitrary mutable
    Python side effects in a checkpointed function; it executes again backward.
    Works even when the input requires no gradient but module parameters do.
    """
    if not enabled or not torch.is_grad_enabled():
        return function(*args, **kwargs)
    if isinstance(function, nn.Module):
        if isinstance(modules, nn.Module):
            modules = (modules, function)
        else:
            modules = (*modules, function)
    batchnorm = _batchnorm_modules(modules)
    return checkpoint(
        function, *args, **kwargs, use_reentrant=False, preserve_rng_state=True,
        context_fn=lambda: (nullcontext(), _recomputation_buffers(batchnorm)),
    )


