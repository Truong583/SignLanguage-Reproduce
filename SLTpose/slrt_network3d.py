import pdb
import copy
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.criterions import SeqKD
from modules import BiLSTMLayer, TemporalConv
from modules.translationPure import SLTModel
from modules.tokenizer import *
from utils.metrics import *
from einops import rearrange
from modules.resnet1d import *
from utils.decodeMax import Decode


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
            torch.nn.Linear(in_features=self.hidden_size, out_features=out_features),
        )
    def forward(self, x):
        return self.mapping(x)

class SLRModel(nn.Module):
    def __init__(
            self, num_classes, c2d_type, conv_type, use_bn=False,
            hidden_size=1024, gloss_dict=None, loss_weights=None,
            weight_norm=True, share_classifier=True, cfg=None, recoder=None
    ):
        super(SLRModel, self).__init__()

        self.loss_weights = loss_weights
        self.gloss_tokenizer = GlossTokenizer_S2G(tokenizer_cfg=cfg['GlossTokenizer'])
        if cfg["datatype"]=="video" or cfg['datatype']=='h5':
            self.points_emb = nn.Linear(cfg['num_key'], 1024)
            self.resnet1D = ResNet1D(1024, 512, kernel_size=2, stride=1, groups=1, n_block=8, n_classes=1000,
                                     downsample_gap=2, increasefilter_gap=4, use_bn=True, use_do=True, verbose=False)
            self.conv1d = TemporalConv(input_size=1024, hidden_size=hidden_size,
                                       conv_type=conv_type, use_bn=use_bn,
                                       num_classes=len(self.gloss_tokenizer))
        self.criterion_init()
        self.num_classes = len(self.gloss_tokenizer)
        self.decoder = Decode(self.gloss_tokenizer, self.num_classes, 'beam', blank_id=self.gloss_tokenizer.pad_id,
                                    beam_width=10)
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, dropout=0.3, bidirectional=True)
        self.classifier = NormLinear(hidden_size, self.num_classes)
        self.translation = SLTModel(input_type='feature', cfg=cfg)
        self.vlembedding = VLEmbeding(hidden_size, self.translation.input_dim)
        assert len(self.translation.text_tokenizer) == self.translation.model.config.vocab_size

        self.recoder = recoder


    def translation_loss(self, feature=None, feature_len=None, text=None, gloss=None):

        tran_dict = self.translation.forward(**{'input_feature': feature, 'input_lengths': feature_len, 'text': text, 'gloss':gloss})
        predtext= self.translation.text_tokenizer.batch_decode(tran_dict['sequences'], skip_special_tokens=True)
        blue = bleu(hypotheses=predtext, references=text)
        if self.recoder:
            self.recoder.wandb_recoder({"train": blue})
        return {'loss':tran_dict['total_loss']}

    def generate(self,feature=None,feature_len=None,gloss=None):

        tran_dict = self.translation.generate(**{'input_feature': feature, 'input_lengths': feature_len,'gloss':gloss})
        return {"decoded_sequences": tran_dict['decoded_sequences']}


    def forward1(self, x, len_x, label=None, label_lgt=None):
        batch, t, n, c = x.shape
        x = rearrange(x, 'b t n c -> b t (n c)')
        x = self.points_emb(x.to(self.points_emb.weight.device))
        x = rearrange(x, 'b t c -> b c t')
        framewise = self.resnet1D(x)


        conv1d_outputs = self.conv1d(framewise, len_x)
        x, lgt = conv1d_outputs['visual_feat'], conv1d_outputs['feat_len'].cpu().int()
        conv1d_outputs['conv_logits'] = self.classifier(conv1d_outputs['visual_feat'])
        tm_outputs_q = self.temporal_model(x,lgt)

        outputs = self.classifier(tm_outputs_q)
        tm_outputs_q = self.vlembedding(tm_outputs_q.permute(1,0,2))

        pred= self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
        conv_pred = None if self.training else self.decoder.decode(conv1d_outputs['conv_logits'], lgt, batch_first=False, probs=False)
        text_pred =None if self.training else self.generate(tm_outputs_q,
                                                            lgt.cpu().int(),
                                                            gloss=pred)['decoded_sequences']

        return {
            "temproal_features":  tm_outputs_q,  "feat_len": lgt,
            "conv_logits": conv1d_outputs['conv_logits'], "sequence_logits": outputs,
            "conv_sents": conv_pred, "recognized_sents": pred, "text_pred":text_pred,
        }

    def forward(self, vid, vid_lgt, gloss, text=None):
        ret_dict = self.forward1(vid, vid_lgt)
        loss = 0

        gloss = self.gloss_tokenizer(gloss)
        label, label_lgt = gloss['gloss_labels'],  gloss['gls_lengths']
        for k, weight in self.loss_weights.items():
            if weight<=0:
                continue
            if k == 'ConvCTC':
                loss1 = weight * self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"],
                                                      label_lgt.cpu().int()).mean()
            elif k == 'SeqCTC':
                loss1 = weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"], #.cpu().int()
                                                      label_lgt.cpu().int()).mean()
            elif k == 'Dist':
                loss1 = weight * self.loss['distillation'](ret_dict["conv_logits"],
                                                           ret_dict["sequence_logits"].detach(),
                                                           use_blank=False)
            elif k== 'Translation':
                output = self.translation_loss(ret_dict['temproal_features'],
                                                       ret_dict['feat_len'].cpu().int(),
                                                       text,
                                                       gloss= ret_dict['recognized_sents'])
                loss1 = weight * output['loss']
            if not np.isinf(loss1.item()) and not np.isnan(loss1.item()):
                loss =loss+ loss1
                if self.recoder:
                    self.recoder.wandb_recoder({"loss": {k: loss1}})
        del ret_dict
        return loss 

    def criterion_init(self):
        self.loss = dict()
        self.loss['CTCLoss'] = torch.nn.CTCLoss(blank=self.gloss_tokenizer.pad_id,reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        return self.loss


if __name__ == '__main__':
    dict_gloss = np.load("./preprocess/phoenix2014/gloss_dict.npy")
    print(dict_gloss)