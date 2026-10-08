import copy
from contextlib import nullcontext

import pytest
import torch
from torch import nn

from repro.activation_checkpoint import checkpoint_call


@pytest.mark.parametrize('checkpointed',[False,True])
def test_independent_graph_groups_match_unchunked_edgeconv_math_in_float64(checkpointed):
    from repro.graphs import EdgeConv,hierarchical_edges
    torch.manual_seed(19)
    layer=EdgeConv(5).double()
    edges=hierarchical_edges(4,4,2,2)
    inputs=torch.randn(37,20,5,dtype=torch.float64)
    records=[]
    for grouped in (False,True):
        layer.zero_grad(set_to_none=True)
        layer.activation_checkpoint=checkpointed
        x=inputs.clone().requires_grad_()
        output=layer(x,edges) if grouped else layer._forward(x,edges)
        output.square().sum().backward()
        records.append((output.detach(),x.grad,[p.grad.clone() for p in layer.parameters()]))
    torch.testing.assert_close(records[0][0],records[1][0],rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(records[0][1],records[1][1],rtol=1e-12,atol=1e-12)
    for a,b in zip(records[0][2],records[1][2]):
        torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)


class ResidualSegment(nn.Module):
    def __init__(self, momentum=.1):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(3, momentum=momentum)
        self.conv2 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(3, momentum=momentum)
        self.drop = nn.Dropout(.3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(self.drop(out)))
        out += x
        return self.relu(out)


@pytest.mark.parametrize('momentum', [.1, None])
@pytest.mark.parametrize('offload_boundaries', [False, True])
def test_segment_matches_gradients_bn_and_rng_with_inplace_residual(momentum, offload_boundaries):
    torch.set_num_threads(2)
    torch.manual_seed(17)
    reference = ResidualSegment(momentum).train()
    actual = copy.deepcopy(reference)
    video = torch.randn(2, 3, 6, 6)
    results = []
    for model, use_checkpoint in ((reference, False), (actual, True)):
        identities = {name: buffer for name, buffer in model.named_buffers()}
        torch.manual_seed(71)
        # Repeated forwards detect cumulative running-statistics drift as well
        # as the first-pass counter increment. Inputs deliberately have no grad.
        for _ in range(2):
            context = (torch.autograd.graph.saved_tensors_hooks(
                lambda tensor: tensor.detach().clone(), lambda tensor: tensor
            ) if offload_boundaries else nullcontext())
            with context:
                output = checkpoint_call(model, video, enabled=use_checkpoint)
                loss = output.square().mean()
            loss.backward()
        assert all(dict(model.named_buffers())[name] is buffer for name, buffer in identities.items())
        results.append((output.detach(), {n:p.grad.clone() for n,p in model.named_parameters()},
                        {n:b.clone() for n,b in model.named_buffers()}, torch.get_rng_state()))
    expected, observed = results
    torch.testing.assert_close(observed[0], expected[0], rtol=0, atol=0)
    for name in expected[1]:
        torch.testing.assert_close(observed[1][name], expected[1][name], rtol=1e-6, atol=1e-7)
    for name in expected[2]:
        torch.testing.assert_close(observed[2][name], expected[2][name], rtol=0, atol=0)
    assert torch.equal(observed[3], expected[3])
    assert actual.bn1.num_batches_tracked.item() == 2


def test_checkpoint_retains_boundaries_instead_of_segment_interiors():
    torch.manual_seed(29)
    model = ResidualSegment().train()
    video = torch.randn(2, 3, 6, 6)
    retained = []
    def pack(tensor):
        retained.append(tensor.numel() * tensor.element_size())
        return tensor.detach().clone()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        out = checkpoint_call(lambda value: model(value), video, modules=(model, model))
    # A checkpoint saves its input and a zero-sized autograd sentinel. The
    # backward graph inside the segment must not retain its many activations.
    assert sum(retained) == video.numel() * video.element_size()
    out.square().sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert model.bn1.num_batches_tracked.item() == 1


def test_recomputation_failure_restores_original_buffers():
    model = ResidualSegment().train()
    calls = [0]
    def segment(value):
        calls[0] += 1
        output = model(value)
        if calls[0] > 1:
            raise RuntimeError('recomputation failed')
        # The extra nonlinear operation ensures the exception occurs before
        # PyTorch's normal early-stop point in the recomputed segment.
        return output.sin()
    original = {name: value for name, value in model.named_buffers()}
    output = checkpoint_call(segment, torch.randn(2, 3, 6, 6), modules=model)
    after_forward = {name: value.clone() for name, value in model.named_buffers()}
    with pytest.raises(RuntimeError, match='recomputation failed'):
        output.square().sum().backward()
    for name, value in model.named_buffers():
        assert value is original[name]
        torch.testing.assert_close(value, after_forward[name], rtol=0, atol=0)


def test_inference_does_not_recompute_or_change_buffers():
    model = ResidualSegment().eval()
    before = {name: value.clone() for name, value in model.named_buffers()}
    with torch.no_grad():
        output = checkpoint_call(model, torch.randn(2, 3, 6, 6))
    assert not output.requires_grad
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
