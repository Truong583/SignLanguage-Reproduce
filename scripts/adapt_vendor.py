"""One-time, deterministic adaptation of copies; upstream stays untouched."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / 'repro/vendor'

def edit(name, before, after):
    p = V / name
    s = p.read_text(encoding='utf-8')
    if before not in s:
        raise ValueError(f'Unexpected upstream content: {name}: {before[:60]}')
    p.write_text(s.replace(before, after), encoding='utf-8')

edit('pos_embed.py', 'dtype=np.float)', 'dtype=np.float64)')
edit('torch_vertex.py', 'from timm.models.layers import DropPath',
     'class DropPath(nn.Identity):\n    def __init__(self, p=0):\n        if p != 0: raise ValueError("Only drop_path=0 is supported")\n        super().__init__()')
edit('torch_edge.py', 'x = F.normalize(x, p=2.0, dim=1)', 'x = F.normalize(x.float(), p=2.0, dim=1)')
edit('torch_edge.py', 'y = F.normalize(y, p=2.0, dim=1)', 'y = F.normalize(y.float(), p=2.0, dim=1)')
edit('backbone.py', 'from modules.gcn_lib.torch_vertex import Grapher, act_layer',
     'from .torch_vertex import Grapher, act_layer')
edit('backbone.py', 'from modules.gcn_lib.temgraph import TemporalGraph',
     'from repro.graphs import TemporalGraph, HierarchicalGraph')
edit('backbone.py', 'def __init__(self, block, layers, num_classes=1000):',
     "def __init__(self, block, layers, num_classes=1000, hsg=True, graph_order='tsg-lsg', graph_conv='edge'):")
edit('backbone.py', 'self.fc = nn.Linear(512 * block.expansion, num_classes)',
     "self.fc = nn.Linear(512 * block.expansion, num_classes)\n        self.hsg = hsg\n        self.graph_order = graph_order\n        if hsg:\n            self.hsg1 = HierarchicalGraph(128, 256, graph_conv=graph_conv)\n            self.hsg2 = HierarchicalGraph(256, 512, graph_conv=graph_conv)")
edit('backbone.py', 'in_channels=256, drop_path=0)', 'in_channels=256, drop_path=0, graph_conv=graph_conv)')
edit('backbone.py', 'in_channels=512, drop_path=0)', 'in_channels=512, drop_path=0, graph_conv=graph_conv)')
edit('backbone.py', 'x = self.layer3(x)', 'high1 = rearrange(x, "N C T H W -> (N T) C H W")\n        x = self.layer3(x)')
edit('backbone.py', 'x = x + self.localG(x) * self.alpha[0]\n        x = x + self.temporalG(x, N) * self.alpha[1]',
     "if self.graph_order == 'tsg-lsg':\n            x = x + self.temporalG(x, N) * self.alpha[1]\n            x = x + self.localG(x) * self.alpha[0]\n        else:\n            x = x + self.localG(x) * self.alpha[0]\n            x = x + self.temporalG(x, N) * self.alpha[1]\n        if self.hsg: x = self.hsg1(high1, x)\n        high2 = x")
edit('backbone.py', 'x = x + self.localG2(x) * self.alpha[2]\n        x = x + self.temporalG2(x, N) * self.alpha[3]',
     "if self.graph_order == 'tsg-lsg':\n            x = x + self.temporalG2(x, N) * self.alpha[3]\n            x = x + self.localG2(x) * self.alpha[2]\n        else:\n            x = x + self.localG2(x) * self.alpha[2]\n            x = x + self.temporalG2(x, N) * self.alpha[3]\n        if self.hsg: x = self.hsg2(high2, x)")
edit('tconv.py', 'torch.div(feat_len, 2)', "torch.div(feat_len, int(ks[1]), rounding_mode='floor')")
sources = {p.name: str(p.relative_to(ROOT)).replace('\\', '/') for p in (ROOT/'MixSignGraph/modules/gcn_lib').glob('*.py') if (V/p.name).exists()}
sources.update({'backbone.py':'MixSignGraph/modules/resnet.py', **{n:f'MixSignGraph/modules/{n}' for n in ['tconv.py','BiLSTM.py','criterions.py']}})
(V/'vendor_manifest.json').write_text(json.dumps({n:{'source':s,'upstream_sha256':hashlib.sha256((ROOT/s).read_bytes()).hexdigest(),'adapted_sha256':hashlib.sha256((V/n).read_bytes()).hexdigest()} for n,s in sources.items()},indent=2),encoding='utf-8')
