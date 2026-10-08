"""Corrected sparse temporal graph and Eq. (1)-(2) hierarchical graph.

These are reconstruction choices, not recovered author code. Graph indexing is
frame-major throughout. No edges may cross videos or nonadjacent frames.
"""
import torch
from torch import nn
from .activation_checkpoint import checkpoint_call


class GCN(nn.Module):
    """Symmetric degree-normalized GCN with self loops, as in GCNConv."""
    def __init__(self, channels):
        super().__init__()
        self.lin = nn.Linear(channels, channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, x, edges):
        # x: [independent graphs, nodes, channels]; edges shared across graphs.
        n = x.shape[-2]
        edges = edges[:, edges[0] != edges[1]]
        loops = torch.arange(n, device=x.device)
        src = torch.cat([edges[0], loops])
        dst = torch.cat([edges[1], loops])
        deg = torch.bincount(dst, minlength=n).to(torch.float32)
        z = self.lin(x)
        norm = (deg[src] * deg[dst]).rsqrt().to(z.dtype)
        out = torch.zeros_like(z)
        out.index_add_(-2, dst, z.index_select(-2, src) * norm[:, None])
        return out + self.bias.to(out.dtype)


class EdgeConv(nn.Module):
    """Sparse max EdgeConv: h(x_i, x_j-x_i); isolated nodes output zero.

    MLP Linear(2C,C)+ReLU is an explicit reconstruction choice; the paper
    identifies EdgeConv but the released HSG/TSG files use GCNConv instead.
    """
    def __init__(self, channels):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2*channels,channels),nn.ReLU())

    def forward(self, x, edges):
        # Batch axis contains independent graphs, unlike the node axis of TSG.
        # Keep the same grouping in every storage mode and in evaluation: CUDA
        # GEMM rounding can depend on shape, affecting downstream kNN ties.
        if x.shape[0]>16:
            return torch.cat([checkpoint_call(self._forward,part,edges,
                modules=self.mlp,enabled=getattr(self,'activation_checkpoint',False) and self.training)
                for part in x.split(16,dim=0)],dim=0)
        return self._forward(x,edges)

    def _forward(self, x, edges):
        src,dst = edges
        center = x.index_select(-2,dst)
        neighbor = x.index_select(-2,src)
        inputs = torch.cat([center,neighbor-center],dim=-1)
        del center,neighbor
        messages = self.mlp(inputs)
        del inputs
        output = torch.full_like(x,-torch.inf,dtype=messages.dtype)
        index = dst[None,:,None].expand(x.shape[0],-1,x.shape[-1])
        output = output.scatter_reduce(1,index,messages,reduce='amax',include_self=True)
        return torch.where(torch.isfinite(output),output,torch.zeros_like(output))


def graph_layer(channels, kind):
    if kind=='edge': return EdgeConv(channels)
    if kind=='gcn': return GCN(channels)
    raise ValueError(f'Unknown graph convolution: {kind}')


def hierarchical_edges(hh, wh, hl, wl, device=None):
    if hh % hl or wh % wl or hh // hl != wh // wl:
        raise ValueError('HSG needs equal integer spatial scale in both axes')
    high = torch.arange(hh * wh, device=device)
    scale = hh // hl
    low = (high // wh // scale) * wl + (high % wh // scale) + hh * wh
    return torch.stack([torch.cat([high, low]), torch.cat([low, high])])


class HierarchicalGraph(nn.Module):
    def __init__(self, high_channels, low_channels, graph_conv='edge'):
        super().__init__()
        self.project = nn.Sequential(nn.Conv2d(high_channels, low_channels, 1),
                                     nn.BatchNorm2d(low_channels), nn.ReLU())
        self.gcn = graph_layer(low_channels,graph_conv)
        self.merge = nn.Sequential(nn.Conv2d(low_channels, low_channels, 3, 2, 1),
                                   nn.BatchNorm2d(low_channels), nn.ReLU())

    def forward(self, high, low):
        high = self.project(high)
        b, c, hh, wh = high.shape
        hl, wl = low.shape[-2:]
        if (hh, wh) != (2 * hl, 2 * wl):
            raise ValueError('Reconstructed HSG uses scale=2; use input size 224')
        edges = hierarchical_edges(hh, wh, hl, wl, high.device)
        z = torch.cat([high.flatten(2), low.flatten(2)], -1).transpose(1, 2)
        z = self.gcn(z, edges).transpose(1, 2)
        hi = z[:, :, :hh * wh].reshape(b, c, hh, wh)
        lo = z[:, :, hh * wh:].reshape(b, c, hl, wl)
        return lo + self.merge(hi)


@torch.no_grad()
def temporal_edges(x, k):
    """Global top-k PAIRS for each adjacent frame, not k edges per node."""
    t, n, c = x.shape
    if t < 2:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)
    # FP32 distances avoid half precision cdist support/overflow problems.
    selections=[]
    for start in range(0,t-1,8):
        end=min(t-1,start+8)
        distance=torch.cdist(x[start:end].float(),x[start+1:end+1].float()).flatten(1)
        selections.append(distance.topk(min(k,n*n),largest=False,dim=-1).indices)
    pairs=torch.cat(selections)
    offsets = torch.arange(t - 1, device=x.device)[:, None] * n
    src, dst = (pairs // n + offsets).flatten(), (pairs % n + offsets + n).flatten()
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])


class TemporalGraph(nn.Module):
    def __init__(self, in_channels, k=49, drop_path=0, graph_conv='edge'):
        super().__init__()
        self.k = k
        self.down_conv = nn.Sequential(nn.Conv3d(in_channels, in_channels, (3, 1, 1),
                                                 padding=(1, 0, 0), bias=False),
                                       nn.BatchNorm3d(in_channels))
        self.up_conv = nn.Sequential(nn.Conv3d(in_channels, in_channels, (3, 1, 1),
                                               padding=(1, 0, 0), bias=False),
                                     nn.BatchNorm3d(in_channels))
        self.gconv = graph_layer(in_channels,graph_conv)

    def forward(self, x, batch):
        bt, c, h, w = x.shape
        t = bt // batch
        z = self.down_conv(x.reshape(batch, t, c, h, w).transpose(1, 2))
        z = z.permute(0, 2, 3, 4, 1).reshape(batch, t, h*w, c)
        results = []
        for sample in z:
            edges = temporal_edges(sample, self.k)
            results.append(self.gconv(sample.reshape(1, t*h*w, c), edges)[0])
        z = torch.stack(results).reshape(batch, t, h, w, c).permute(0, 4, 1, 2, 3)
        return self.up_conv(z).transpose(1, 2).reshape(bt, c, h, w)
