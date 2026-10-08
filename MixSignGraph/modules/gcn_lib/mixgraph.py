from torch_geometric.nn import GCNConv, SAGEConv, EdgeConv
import torch.nn as nn
import torch
from timm.models.layers import DropPath
from einops import rearrange
import numpy as np
# from .pos_embed import get_2d_relative_pos_embed
import random


class MixGraph(nn.Module):
    def __init__(self, in_channels, k=4, drop_path=0.0, hw1=14, hw2=7):
        super(MixGraph, self).__init__()
        # inchannles =[channel1, channel2]
        self.k = k
        self.n1, self.n2 = hw1 * hw1, hw2 * hw2
        self.scale = hw1 // hw2
        self.reduction_channel = in_channels[1]
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels[0], self.reduction_channel, 1, stride=1, padding=0),
            nn.BatchNorm2d(self.reduction_channel),
            nn.ReLU(inplace=True)
        )
        self.up_conv = nn.Sequential(
            nn.Conv2d(self.reduction_channel, self.reduction_channel, 3, stride=self.scale, padding=1),
            nn.BatchNorm2d(self.reduction_channel),
            nn.ReLU(inplace=True)
        )
        self.gconv = GCNConv(self.reduction_channel, self.reduction_channel)

    # def build_egde(self,b, t, h1, w1, h2, w2):
    #     h_index = torch.arange(self.n1).view(1,-1).expand(b*t,self.n1).view(b,t,h1,w1)
    #     w_index = (h_index//h1//self.scale*h2+h_index%h1//self.scale).int()
    #     h_index = h_index.view(b, t, h1*w1)
    #     w_index = w_index.view(b, t, h1*w1)
    #     print(h_index, w_index)
    #     h_index_off = torch.arange(h1*w1,t*h1*w1, step=h1*w1).unsqueeze(1).expand(t, h1*w1)
    # #

    #     # print(w_index)

    def build_egde(self, t, h1, w1, h2, w2):
        h_index = torch.arange(self.n1).view(1, -1).expand(t, self.n1).view(t, h1, w1)
        w_index = (h_index // h1 // self.scale * h2 + h_index % h1 // self.scale).int()
        h_index = h_index.view(t, h1 * w1)
        w_index = w_index.view(t, h1 * w1)
        # print(h_index, w_index)
        h_index_off = torch.arange(0, t * h1 * w1, step=h1 * w1).unsqueeze(1).expand(t, h1 * w1)
        h_index = h_index + h_index_off
        w_index_off = torch.arange(0, t * h2 * w2, step=h2 * w2).unsqueeze(1).expand(t, h2 * w2).repeat(1,
                                                                                                        self.scale * self.scale)
        # print(w_index.shape, w_index_off.shape, h1, w1, h2, w2, self.scale)
        w_index = w_index + w_index_off + t * h1 * w1
        # w_index_off = torch.arange(0,t*h1*w1, step=h1*w1).unsqueeze(1).expand(t, h1*w1)

        finaledge = torch.stack((h_index.reshape(-1), w_index.reshape(-1)), dim=-1)
        finaledge_re = torch.stack((h_index.reshape(-1), w_index.reshape(-1)), dim=-1)
        finaledge = torch.cat((finaledge, finaledge_re), dim=0).permute(1, 0)
        return finaledge

    def forward(self, featureH, featureL, batch):
        # b*t, c1, h1, w1,  feautre1
        # b*t, c2, h2, w2, feautre2
        bt, c1, h1, w1 = featureH.shape
        bt, c2, h2, w2 = featureL.shape
        featureH = self.down_conv(featureH)  # b*t, c2, h1, w1
        featureH = featureH.view(batch, bt // batch, c2, h1 * w1)
        featureL = featureL.view(batch, bt // batch, c2, h2 * w2)

        out = torch.zeros(batch, bt // batch, c2, h1 * w1 + h2 * w2).to(featureH.device)
        finnalegde = self.build_egde(bt // batch, h1, w1, h2, w2).to(featureH.device)
        for i in range(batch):
            feature = torch.cat((featureH[i], featureL[i]), dim=-1)
            t, c, n = feature.shape
            feature = rearrange(feature, 't c n-> (t n) c')
            feature = self.gconv(feature, finnalegde)
            out[i] = feature.view(t, n, c).permute(0, 2, 1)
        featureHO, featureLO = out[:, :, :, :h1 * w1], out[:, :, :, h1 * w1:]
        # print(featureHO.shape, featureLO.shape)
        featureHO, featureLO = featureHO.view(bt, c2, h1, w1), featureLO.view(bt, c2, h2, w2)
        featureHO = self.up_conv(featureHO)
        # print(featureHO.shape, featureLO.shape)
        return featureHO + featureLO


if __name__ == '__main__':
    model = MixGraph([256, 512], k=4, drop_path=0.0, hw1=28, hw2=7)
    torch.manual_seed(1)
    seed = 2
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    x = torch.rand([4 * 200, 256, 28, 28])
    xl = torch.rand([4 * 200, 512, 7, 7])
    y = model(x, xl, 4)
    print(y.shape)


