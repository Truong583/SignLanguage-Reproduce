"""Paper-scoped ablations. Unspecified architecture choices are logged in the suite.

All graph indices are frame-major; a temporal edge joins adjacent frames only.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .graphs import EdgeConv, GCN, HierarchicalGraph, TemporalGraph, temporal_edges
from .vendor.backbone import ResNet, BasicBlock
from .vendor.torch_vertex import Grapher


def distances(x, y, kind="euclidean"):
    x, y = x.float(), y.float()
    if kind == "cosine":
        return 1 - F.normalize(x, dim=-1) @ F.normalize(y, dim=-1).transpose(-1, -2)
    if kind == "chebyshev":
        return torch.cdist(x, y, p=float("inf"))
    if kind == "euclidean":
        return torch.cdist(x, y)
    raise ValueError(f"Unknown distance: {kind}")


@torch.no_grad()
def neighbors(x, k, kind="euclidean", sparse=False):
    n = len(x)
    d = distances(x, x, kind)
    if sparse:
        pairs = d.flatten().topk(min(k, n*n), largest=False).indices
        return torch.stack([pairs % n, pairs // n])
    ids = d.topk(min(k, n), largest=False).indices
    dst = torch.arange(n, device=x.device)[:, None].expand_as(ids)
    return torch.stack([ids.flatten(), dst.flatten()])


def drop_edges(edges, rate, training):
    if not 0 <= rate < 1:
        raise ValueError("DropEdge rate must lie in [0,1)")
    if rate and training:
        edges = edges[:, torch.rand(edges.shape[1], device=edges.device) >= rate]
    return edges


class SAGE(nn.Module):
    def __init__(self, c, out=None):
        super().__init__()
        out=out or c
        self.root = nn.Linear(c, out)
        self.neighbor = nn.Linear(c, out, bias=False)

    def forward(self, x, edges):
        src, dst = edges
        z = torch.zeros_like(x)
        z.index_add_(1, dst, x[:, src])
        degree = torch.bincount(dst, minlength=x.shape[1]).clamp_min(1).to(x.dtype)
        return self.root(x) + self.neighbor(z / degree[None, :, None])


class GATv2(nn.Module):
    """Single-head GATv2 with separate source/target projections and self loops."""
    def __init__(self, c, out=None):
        super().__init__()
        out=out or c
        self.source = nn.Linear(c, out, bias=False)
        self.target = nn.Linear(c, out, bias=False)
        self.attention = nn.Parameter(torch.empty(out))
        self.bias = nn.Parameter(torch.zeros(out))
        nn.init.xavier_uniform_(self.attention[None])

    def forward(self, x, edges):
        loops = torch.arange(x.shape[1], device=x.device)
        edges = edges[:, edges[0] != edges[1]]
        src = torch.cat([edges[0], loops]); dst = torch.cat([edges[1], loops])
        value = self.source(x)
        score = (F.leaky_relu(value[:, src] + self.target(x)[:, dst], .2) * self.attention).sum(-1)
        index = dst[None].expand(x.shape[0], -1)
        maximum = score.new_full(x.shape[:2], -torch.inf).scatter_reduce(1, index, score, reduce="amax", include_self=True)
        exp = (score - maximum.gather(1, index)).exp()
        denominator = torch.zeros_like(maximum).scatter_add_(1, index, exp)
        weights = exp / denominator.gather(1, index).clamp_min(torch.finfo(exp.dtype).tiny)
        out = torch.zeros_like(value)
        out.index_add_(1, dst, value[:, src] * weights.to(value.dtype)[..., None])
        return out + self.bias.to(out.dtype)


class LocalGCN(nn.Module):
    def __init__(self,c,out):
        super().__init__(); self.lin=nn.Linear(c,out,bias=False)
        self.bias=nn.Parameter(torch.zeros(out))
    def forward(self,x,edges):
        n=x.shape[1]; ids=torch.arange(n,device=x.device)
        edges=edges[:,edges[0]!=edges[1]]
        src=torch.cat([edges[0],ids]); dst=torch.cat([edges[1],ids])
        degree=torch.bincount(dst,minlength=n).float()
        values=self.lin(x)
        weights=(degree[src]*degree[dst]).rsqrt().to(values.dtype)
        result=torch.zeros_like(values); result.index_add_(1,dst,values[:,src]*weights[None,:,None])
        return result+self.bias.to(result.dtype)


def layer(c, kind):
    if kind == "edge": return EdgeConv(c)
    if kind == "gcn": return GCN(c)
    if kind == "sage": return SAGE(c)
    if kind == "gatv2": return GATv2(c)
    raise ValueError(kind)


class Spatial(nn.Module):
    def __init__(self, c, k, nodes, cfg):
        super().__init__()
        self.k, self.distance = k, cfg.get("distance", "euclidean")
        self.sparse = cfg.get("lsg_type", "dense") == "sparse"
        self.drop_rate = cfg.get("drop_edge", 0.)
        self.sa = cfg.get("backbone", "resnet18") == "sa"
        # Preserve the released LSG parameterization for the reference setting.
        self.legacy = not self.sa and not self.sparse and not self.drop_rate and self.distance == "euclidean" and cfg.get("graph_conv", "edge") == "edge"
        self.graph = Grapher(c, k, 1, "edge", "relu", "batch", True, False, 0., 1, nodes, 0., True)
        if self.legacy: return
        if self.sa:
            self.attn = nn.MultiheadAttention(c, 8, batch_first=True)
            self.project=nn.Linear(c,2*c)
            self.norm=nn.Sequential(nn.BatchNorm2d(2*c),nn.ReLU())
        else:
            kind=cfg.get('graph_conv','edge')
            if kind=='edge': self.edge_mlp=self.graph.graph_conv.gconv.nn
            else:
                self.node_layer={'gcn':LocalGCN,'sage':SAGE,'gatv2':GATv2}[kind](c,2*c)
                self.norm=nn.Sequential(nn.BatchNorm2d(2*c),nn.ReLU())
        del self.graph.graph_conv

    def forward(self, x):
        if self.legacy: return self.graph(x)
        b,c,h,w=x.shape; n=h*w
        z = self.graph.fc1(x).flatten(2).transpose(1, 2)
        if self.sa:
            z = self.project(self.attn(z, z, z, need_weights=False)[0])
            z=self.norm(z.transpose(1,2).reshape(b,2*c,h,w))
        else:
            with torch.no_grad():
                normalized=F.normalize(z.float(),dim=-1)
                bias=self.graph._get_relative_pos(self.graph.relative_pos,h,w)[0]
                edges=[]
                for sample in normalized:
                    d=distances(sample,sample,self.distance)
                    if self.distance=='euclidean': d=d.square()
                    d=d+bias
                    if self.sparse:
                        d.fill_diagonal_(float('inf'))
                        ids=d.flatten().topk(min(self.k,n*(n-1)),largest=False).indices
                        edges.append(torch.stack([ids%n,ids//n]))
                    else:
                        ids=d.topk(min(self.k,n),largest=False).indices
                        dst=torch.arange(n,device=x.device)[:,None].expand_as(ids)
                        edges.append(torch.stack([ids.flatten(),dst.flatten()]))
            if hasattr(self,'edge_mlp'):
                center=torch.stack([sample[e[1]] for sample,e in zip(z,edges)])
                neighbor=torch.stack([sample[e[0]] for sample,e in zip(z,edges)])
                messages=self.edge_mlp(torch.cat([center,neighbor-center],-1).transpose(1,2).unsqueeze(-1)).squeeze(-1).transpose(1,2)
                destination=torch.stack([e[1] for e in edges])
                if self.drop_rate and self.training:
                    mask=torch.rand(destination.shape,device=x.device)>=self.drop_rate
                    messages=messages.masked_fill(~mask[...,None],-torch.inf)
                result=torch.full((b,n,2*c),-torch.inf,device=x.device,dtype=messages.dtype)
                index=destination[...,None].expand_as(messages)
                result=result.scatter_reduce(1,index,messages,reduce='amax',include_self=True)
                z=torch.where(torch.isfinite(result),result,torch.zeros_like(result)).transpose(1,2).reshape(b,2*c,h,w)
            else:
                result=torch.stack([self.node_layer(sample[None],e)[0] for sample,e in zip(z,edges)])
                z=self.norm(result.transpose(1,2).reshape(b,2*c,h,w))
        return self.graph.fc2(z)


class Temporal(TemporalGraph):
    def __init__(self, c, k, cfg):
        super().__init__(c, k, graph_conv="edge")
        self.gconv = layer(c, cfg.get("graph_conv", "edge"))
        self.distance = cfg.get("distance", "euclidean")
        self.dense = cfg.get("tsg_type", "sparse") == "dense"
        self.sa = cfg.get("backbone", "resnet18") == "sa"
        if self.sa:
            del self.gconv
            self.attn = nn.MultiheadAttention(c, 8, batch_first=True)

    def forward(self, x, batch):
        bt, c, h, w = x.shape; t = bt // batch; n = h*w
        z = self.down_conv(x.reshape(batch, t, c, h, w).transpose(1, 2))
        z = z.permute(0, 2, 3, 4, 1).reshape(batch, t, n, c)
        results = []
        for sample in z:
            if self.sa:
                # Pairwise attention keeps the adjacency-in-time constraint.
                out = torch.zeros_like(sample); counts = sample.new_zeros((t, 1, 1))
                for index in range(t-1):
                    pair = sample[index:index+2].reshape(1, 2*n, c)
                    out[index:index+2] += self.attn(pair, pair, pair, need_weights=False)[0].reshape(2, n, c)
                    counts[index:index+2] += 1
                results.append((out / counts.clamp_min(1)).reshape(t*n, c))
                continue
            if self.distance == "euclidean" and not self.dense:
                edges = temporal_edges(sample, self.k)
            else:
                parts = []
                with torch.no_grad():
                    for index in range(t-1):
                        d = distances(sample[index], sample[index+1], self.distance)
                        if self.dense:
                            ids = d.topk(min(self.k, n), largest=False).indices
                            src = torch.arange(n, device=x.device)[:, None].expand_as(ids).flatten() + index*n
                            dst = ids.flatten() + (index+1)*n
                        else:
                            ids = d.flatten().topk(min(self.k, n*n), largest=False).indices
                            src, dst = ids//n + index*n, ids%n + (index+1)*n
                        parts.append(torch.stack([torch.cat([src, dst]), torch.cat([dst, src])]))
                edges = torch.cat(parts, 1) if parts else torch.empty(2, 0, dtype=torch.long, device=x.device)
            if getattr(self,'capture_edges',False): self.recorded_edges=edges.detach().cpu()
            results.append(self.gconv(sample.reshape(1, t*n, c), edges)[0])
        z = torch.stack(results).reshape(batch, t, h, w, c).permute(0, 4, 1, 2, 3)
        return self.up_conv(z).transpose(1, 2).reshape_as(x)


class Hierarchical(HierarchicalGraph):
    def __init__(self, high, low, k, cfg):
        super().__init__(high, low, graph_conv="edge")
        self.gcn = layer(low, cfg.get("graph_conv", "edge"))
        self.k, self.distance = k, cfg.get("distance", "euclidean")
        self.dynamic = cfg.get("hsg_type", "fixed") == "dynamic"
        self.drop_rate = cfg.get("drop_edge", 0.)
        self.sa = cfg.get("backbone", "resnet18") == "sa"
        if self.sa:
            del self.gcn
            self.attn = nn.MultiheadAttention(low, 8, batch_first=True)

    def forward(self, high, low):
        from .graphs import hierarchical_edges
        high = self.project(high)
        b, c, hh, wh = high.shape; hl, wl = low.shape[-2:]
        fixed = hierarchical_edges(hh, wh, hl, wl, high.device)
        z = torch.cat([high.flatten(2), low.flatten(2)], -1).transpose(1, 2)
        if self.sa:
            z = self.attn(z, z, z, need_weights=False)[0]
        else:
            results = []
            for sample in z:
                edges = neighbors(sample, self.k, self.distance) if self.dynamic else fixed
                edges = drop_edges(edges, self.drop_rate, self.training)
                results.append(self.gcn(sample[None], edges)[0])
            z = torch.stack(results)
        z = z.transpose(1, 2)
        hi = z[:, :, :hh*wh].reshape(b, c, hh, wh)
        lo = z[:, :, hh*wh:].reshape(b, c, hl, wl)
        return lo + self.merge(hi)


class AblationBackbone(ResNet):
    def __init__(self, cfg):
        super().__init__(BasicBlock, [2, 2, 2, 2], hsg=False)
        # Remove disabled graph parameters so DDP cannot wait on unused modules.
        for name in ("localG", "localG2", "temporalG", "temporalG2", "alpha"):
            delattr(self, name)
        self.fc = nn.Identity()
        self.blocks = nn.ModuleDict()
        self.scales = cfg.get("graph_scales", [16, 32])
        masks = cfg.get("module_mask", [True]*6)
        if len(masks) != 6: raise ValueError("module_mask must have six booleans: L1,T1,H1,L2,T2,H2")
        order = cfg.get("module_order", "tsg-lsg-hsg").split("-")
        if sorted(order) != ["hsg", "lsg", "tsg"]: raise ValueError("Invalid module_order")
        self.order = order
        self.gains = nn.ParameterDict()
        channels = {4:(64,64), 8:(64,128), 16:(128,256), 32:(256,512)}
        lk, tk = cfg.get("local_k", [3,4]), cfg.get("temporal_k", [49,49])
        for scale in self.scales:
            high, low = channels[scale]; index = 1 if scale == 32 else 0
            enabled = masks[index*3:index*3+3]
            for kind, active in zip(("lsg", "tsg", "hsg"), enabled):
                if not active: continue
                key = f"s{scale}_{kind}"
                if kind == "lsg": block = Spatial(low, lk[index], (224//scale)**2, cfg)
                elif kind == "tsg": block = Temporal(low, tk[index], cfg)
                else: block = Hierarchical(high, low, lk[index], cfg)
                self.blocks[key] = block
                if kind != "hsg": self.gains[key] = nn.Parameter(torch.ones(()))

    def forward(self, video):
        b, _, t, _, _ = video.shape
        x = self.relu(self.bn1(self.conv1(video)))
        high = x.transpose(1, 2).flatten(0, 1)
        x = self.maxpool(x)
        for scale, stage in zip((4, 8, 16, 32), (self.layer1, self.layer2, self.layer3, self.layer4)):
            x = stage(x); _, c, _, h, w = x.shape
            low = x.transpose(1, 2).reshape(b*t, c, h, w)
            for kind in self.order:
                key = f"s{scale}_{kind}"
                if key not in self.blocks: continue
                block = self.blocks[key]
                if kind == "hsg": low = block(high, low)
                else: low = low + self.gains[key] * (block(low, b) if kind == "tsg" else block(low))
            high = low
            x = low.reshape(b, t, c, h, w).transpose(1, 2)
        return F.adaptive_avg_pool2d(low, 1).flatten(1)


class PVIGFFN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Conv2d(c, 4*c, 1), nn.BatchNorm2d(4*c))
        self.act = nn.GELU()
        self.fc2 = nn.Sequential(nn.Conv2d(4*c, c, 1), nn.BatchNorm2d(c))
        self.drop_path = nn.Identity()

    def forward(self, x): return x + self.fc2(self.act(self.fc1(x)))


class PVIGTiny(nn.Module):
    """Compatible feature trunk of the official pvig_ti_224_gelu ImageNet release.

    Tiny is an explicit assumption: MixSignGraph does not identify the PyViG size.
    Module names/shapes match the pinned Huawei release for strict loading.
    """
    def __init__(self):
        super().__init__()
        self.stem = nn.Module()
        self.stem.convs = nn.Sequential(nn.Conv2d(3,24,3,2,1), nn.BatchNorm2d(24), nn.GELU(),
            nn.Conv2d(24,48,3,2,1), nn.BatchNorm2d(48), nn.GELU(), nn.Conv2d(48,48,3,1,1), nn.BatchNorm2d(48))
        self.pos_embed = nn.Parameter(torch.zeros(1,48,56,56))
        blocks = []; previous = 48; index = 0; nodes = 56*56
        for stage, (c, depth, r) in enumerate(zip((48,96,240,384), (2,2,6,2), (4,2,1,1))):
            if stage:
                down = nn.Module(); down.conv = nn.Sequential(nn.Conv2d(previous,c,3,2,1), nn.BatchNorm2d(c))
                # Sequential executes children whereas a plain Module has no forward.
                down.forward = down.conv.forward
                blocks.append(down); nodes //= 4
            for _ in range(depth):
                blocks.append(nn.Sequential(Grapher(c,9,min(index//4+1,5),"mr","gelu","batch",True,False,.2,r,nodes,0.,True),PVIGFFN(c)))
                index += 1
            previous = c
        self.backbone = nn.Sequential(*blocks)
        self.prediction = nn.Sequential(nn.Conv2d(384,1024,1),nn.BatchNorm2d(1024),nn.GELU(),nn.Dropout(0.),nn.Conv2d(1024,1000,1))

    def forward(self, video):
        x = video.transpose(1,2).flatten(0,1)
        x = self.stem.convs(x) + self.pos_embed
        return F.adaptive_avg_pool2d(self.backbone(x), 1).flatten(1)


class SwinFrames(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import swin_t
        self.model = swin_t(weights=None)

    def forward(self, video):
        x = video.transpose(1,2).flatten(0,1)
        return self.model.avgpool(self.model.permute(self.model.norm(self.model.features(x)))).flatten(1)
