from torch_geometric.nn import GCNConv, SAGEConv, EdgeConv
import torch.nn as nn
import torch
from einops import rearrange
import numpy as np
import random
import torch.nn.functional as F
from .pos_embed import get_2d_relative_pos_embed
import torch.nn.functional as F
from timm.models.layers import DropPath

from torch.nn import Sequential as Seq, Linear as Lin, Conv2d


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer

    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


class BasicConv(nn.Sequential):
    def __init__(self, channels, act='relu', norm=None, bias=True, drop=0.):
        m = []
        for i in range(1, len(channels)):
            m.append(Conv2d(channels[i - 1], channels[i], 1, bias=bias, groups=4))
            if norm is not None and norm.lower() != 'none':
                m.append(norm_layer(norm, channels[-1]))
            if act is not None and act.lower() != 'none':
                m.append(act_layer(act))
            if drop > 0:
                m.append(nn.Dropout2d(drop))

        super(BasicConv, self).__init__(*m)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()




def EucDis(x, y):
    '''
        Args:
            x =[b c h w]
            x: tensor (batch_size, num_points, num_dims)
        Returns:
            pairwise distance: (batch_size, num_points, num_points)
    '''
    # x =[b, c, h w], y=[b, c,h w]
    x =rearrange(x, "b c h w-> b (h w) c")
    return torch.einsum('bnc,bkc->bnk', x, x)


def ForEucDis(x, y):
    with torch.no_grad():
        b, c ,h, w = x.shape
        x = rearrange(x, "b c h w-> b (h w) c")
        sim = torch.cdist(x, x)
        mask = torch.triu(torch.ones(b, h*w, h*w), diagonal=0).to(x.device)
    return sim*mask


class LocalGraph(nn.Module):
    def __init__(self, in_channels, k=4, drop_path=0.0,conv_type='gcn', relative_pos=True, n=49):
        super(LocalGraph, self).__init__()
        self.reduction_channel = in_channels
        r = 1
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.reduction_channel, 1, stride=1, padding=0),
            nn.BatchNorm2d(self.reduction_channel),
        )
        self.up_conv = nn.Sequential(
            nn.Conv2d(self.reduction_channel, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        if conv_type == 'edge':
            self.gconv = EdgeConv(nn.Linear(self.reduction_channel, self.reduction_channel))
        elif conv_type == 'SAGE'.lower():
            self.gconv = SAGEConv(self.reduction_channel, self.reduction_channel)
        else:
            self.gconv = GCNConv(self.reduction_channel, self.reduction_channel)
        # self.graph_conv = EdgeConv(self.reduction_channel, self.reduction_channel)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.relative_pos = None
        if relative_pos:
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels, int(n ** 0.5)))).unsqueeze(
                0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(
                relative_pos_tensor, size=(n, n // (r * r)), mode='bicubic', align_corners=False)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)
        self.k = k

    def _get_relative_pos(self, relative_pos, H, W):
        return relative_pos

    def forward(self, x, batch):
        x = self.down_conv(x)
        edge = self.build_graph(x).to(x.device)
        b, c, h, w = x.shape
        tlen= b//batch
        x = x.view(batch, tlen, c, h, w)
        x = rearrange(x,"b t c h w-> b (t h w) c")
        out = torch.zeros_like(x).to(x.device)
        for i in range(batch):
            out[i] = self.gconv(x[i], edge[i])
        x = out.permute(0, 2, 1).view(batch ,c, tlen, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(b, c, h, w)
        x = self.up_conv(x)
        return x

    # def build_graph(self, x, batch=0):
    #     B, C, H, W = x.shape
    #     relative_pos = self._get_relative_pos(self.relative_pos, H, W)
    #
    #     sim = -(EucDis(x, x) + relative_pos)
    #     b,n, n, =sim.shape
    #     sim = F.normalize(sim.view(B, -1), dim=-1)
    #     sim = torch.where(sim < 0.05, 100, sim)
    #     sim = sim.view(b,n,n)
    #     _, topk_indices = torch.topk(sim, k=self.k)
    #
    #     node_pairs = torch.arange(n).unsqueeze(1).to(sim.device)  # 创建一个列向量，包含0到n-1的索引值
    #     node_pairs = node_pairs.expand(b, n, self.k)
    #     # print(node_pairs)
    #     # print("----"*20)
    #     # print(topk_indices)
    #     topk_indices = (topk_indices+ node_pairs*n).view(b,-1)
    #     # print("----" * 20)
    #     row_indices, col_indices = topk_indices // n, topk_indices % n
    #     t = batch//b
    #     finaledge = torch.zeros((b , n*self.k, 2), dtype=torch.int)
    #     finaledge[:, :, 0] = row_indices
    #     finaledge[:,  :, 1] = col_indices
    #     finaledge_re = torch.stack((finaledge[:, :, 1], finaledge[:, :, 0]), dim=-1)
    #     finaledge = torch.cat((finaledge, finaledge_re), dim=1).permute(0, 2, 1).detach()
    #     return finaledge

    def build_graph(self, x, batch=0):
        B, C, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)

        sim = -(EucDis(x, x) + relative_pos)
        b, n, n, = sim.shape
        sim = F.normalize(sim.view(B, -1), dim=-1)
        sim = torch.where(sim < 0.05, 100, sim)
        sim = sim.view(b, n, n)
        _, topk_indices = torch.topk(sim, k=self.k)

        node_pairs = torch.arange(n).unsqueeze(1).expand(b, n, self.k).to(sim.device)

        topk_indices = (topk_indices + node_pairs * n).view(b, -1)
        # print("----" * 20)
        row_indices, col_indices = topk_indices // n, topk_indices % n

        t = batch // b
        finaledge = torch.zeros((batch,t, n * self.k, 2), dtype=torch.int)
        for i in range(t):
            finaledge[:, i, :, 0] = row_indices[:, i, :] + i * n
            finaledge[:, i, :, 1] = col_indices[:, i, :] + i * n
        finaledge = finaledge.view(b, t*n*self.k, 2)
        finaledge_re = torch.stack((finaledge[:, :, 1], finaledge[:, :, 0]), dim=-1)
        finaledge = torch.cat((finaledge, finaledge_re), dim=1).permute(0, 2, 1).detach()
        return finaledge

    #

'''
       
        finaledge = finaledge.view(b, t_1 * self.k, 2)
        finaledge_re = torch.stack((finaledge[:, :, 1], finaledge[:, :, 0]), dim=-1)
        finaledge = torch.cat((finaledge, finaledge_re), dim=1).permute(0, 2, 1).detach()
        return finaledge
'''

if __name__ == '__main__':
    model = LocalGraph(k=2, in_channels=32, drop_path=0, n=14*14)
    torch.manual_seed(1)
    seed=2
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    x = torch.rand([10,32,14,14])
    # print(x)

    y = model(x, 10)

    print(y.shape)
