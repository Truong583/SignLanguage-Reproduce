import pdb
import copy
import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.criterions import SeqKD
from modules import BiLSTMLayer, TemporalConv
import modules.resnet as resnet
from modules.translation import SLTModel, XentLoss

from torch.cuda.amp import autocast as autocast
from modules.tokenizer import *

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class VLEmbeding(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.hidden_size = out_features
        self.mapping = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=self.hidden_size, out_features=out_features)
        )
    def forward(self, x):
        return self.mapping(x)



class SLRModel(nn.Module):
    def __init__(
            self, num_classes, c2d_type, conv_type, use_bn=False,
            hidden_size=1024, gloss_dict=None, loss_weights=None,
            weight_norm=True, share_classifier=True, cfg=None
    ):
        super(SLRModel, self).__init__()

        self.loss_weights = loss_weights
        self.conv2d = getattr(resnet, c2d_type)()
        self.conv2d.fc = Identity()
        self.gloss_tokenizer = GlossTokenizer_S2G(tokenizer_cfg=cfg['GlossTokenizer'])
        self.criterion_init()
        self.num_classes = len(self.gloss_tokenizer)

        self.conv1d = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,)
        self.decoder = utils.Decode(self.gloss_tokenizer, self.num_classes, 'beam', blank_id=self.gloss_tokenizer.pad_id,
                                    beam_width=10)
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, dropout=0.3, bidirectional=True)

        self.classifier = NormLinear(hidden_size, self.num_classes)

        self.translation = SLTModel(input_type='feature', cfg=cfg)
        self.vlembedding= VLEmbeding(hidden_size, self.translation.input_dim)

    def translation_loss(self, feature=None, feature_len=None, text=None, gloss=None):
        feature = self.vlembedding(feature)
        tran_dict = self.translation.forward(**{'input_feature': feature, 'input_lengths': feature_len, 'text': text, 'gloss':gloss})
        return tran_dict['total_loss']

    def generate(self,feature=None,feature_len=None,gloss=None):
        feature = self.vlembedding(feature)
        tran_dict = self.translation.generate(**{'input_feature': feature, 'input_lengths': feature_len,'gloss':gloss})
        return {"decoded_sequences": tran_dict['decoded_sequences']}

    def forward1(self, x, len_x, label=None, label_lgt=None):
        if len(x.shape) == 5:
            batch, temp, channel, height, width = x.shape
            framewise = self.conv2d(x.permute(0,2,1,3,4)).view(batch, temp, -1).permute(0,2,1) # btc -> bct
        else:
            framewise = x
        conv1d_outputs = self.conv1d(framewise, len_x)
        conv1d_outputs['conv_logits'] = self.classifier(conv1d_outputs['visual_feat'])
        x, lgt = conv1d_outputs['visual_feat'], conv1d_outputs['feat_len'].cpu().int()
        tm_outputs = self.temporal_model(x, lgt)
        outputs = self.classifier(tm_outputs['predictions'])

        pred = None if self.training  else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
        # pred= self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
        conv_pred = None if self.training else self.decoder.decode(conv1d_outputs['conv_logits'], lgt, batch_first=False, probs=False)
        text_pred =None if self.training else self.generate(tm_outputs['predictions'].permute(1, 0, 2),
                                                            lgt.cpu().int(),
                                                            gloss=None)['decoded_sequences']

        return { 
            "temproal_features": tm_outputs['predictions'], "feat_len": lgt,
            "conv_logits": conv1d_outputs['conv_logits'], "sequence_logits": outputs,
            "conv_sents": conv_pred, "recognized_sents": pred, "text_pred":text_pred
        }

    def forward(self, vid, vid_lgt, gloss, text=None):
        ret_dict = self.forward1(vid, vid_lgt)
        gloss = self.gloss_tokenizer(gloss)
        label, label_lgt = gloss['gloss_labels'],  gloss['gls_lengths']
        loss = 0
        for k, weight in self.loss_weights.items():
            if weight<=0:
                continue
            if k == 'ConvCTC':
                loss1 = weight * self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
                if not np.isinf(loss1.item()) and not np.isnan(loss1.item()):
                    loss +=loss1
            elif k == 'SeqCTC':
                loss1 = weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
                if not np.isinf(loss1.item()) and not np.isnan(loss1.item()):
                    loss += loss1
            elif k == 'Dist':
                loss1 = weight * self.loss['distillation'](ret_dict["conv_logits"],
                                                           ret_dict["sequence_logits"].detach(),
                                                           use_blank=False)
                if not np.isinf(loss1.item()) and not np.isnan(loss1.item()):
                    loss += loss1
            elif k== 'Translation':
                loss1 = weight * self.translation_loss(ret_dict['temproal_features'].permute(1,0,2),
                                                       ret_dict['feat_len'].cpu().int(),
                                                       text,
                                                       gloss= ret_dict['recognized_sents'])
                if not np.isinf(loss1.item()) and not np.isnan(loss1.item()):
                    loss += loss1
        return loss

    def criterion_init(self):
        self.loss = dict()
        self.loss['CTCLoss'] = torch.nn.CTCLoss(blank=self.gloss_tokenizer.pad_id,reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        return self.loss


if __name__ == '__main__':
    dict_gloss = np.load("./preprocess/phoenix2014/gloss_dict.npy")
    print(dict_gloss)