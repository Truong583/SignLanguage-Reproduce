import math
import torch
from torch import nn
from torch.nn import functional as F
from .vendor.backbone import ResNet, BasicBlock
from .vendor.tconv import TemporalConv
from .vendor.BiLSTM import BiLSTMLayer
from .vendor.criterions import SeqKD
from .data import encode_targets


def load_translation(path,lang):
    from transformers import MBartForConditionalGeneration, MBartTokenizer
    tokenizer=MBartTokenizer.from_pretrained(path,src_lang=lang,tgt_lang=lang,local_files_only=True)
    model=MBartForConditionalGeneration.from_pretrained(path,local_files_only=True,dropout=.3,attention_dropout=.1)
    return tokenizer,model


class NormLinear(nn.Module):
    def __init__(self, in_features, classes):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, classes))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        return x @ F.normalize(self.weight, dim=0)


class SignModel(nn.Module):
    def __init__(self, cfg, vocab, initialize=True):
        super().__init__()
        self.cfg, self.vocab = cfg, vocab
        self.activation_offload=cfg.get('activation_offload','cpu_checkpoint')
        if self.activation_offload not in ('none','cpu','cpu_checkpoint'):
            raise ValueError('activation_offload must be none, cpu or cpu_checkpoint')
        self.input_kind = cfg.get('input_kind','rgb')
        hidden = cfg.get('hidden_size',1024)
        if self.input_kind == 'rgb':
            backbone_kind = cfg.get('backbone','resnet18')
            if backbone_kind in ('swin_t','pvig_tiny'):
                from .ablation import SwinFrames, PVIGTiny
                self.backbone = SwinFrames() if backbone_kind=='swin_t' else PVIGTiny()
                input_dim = 768 if backbone_kind=='swin_t' else 384
                if initialize and not cfg.get('allow_random_init',False):
                    weights = torch.load(cfg['backbone_weights'], map_location='cpu', weights_only=True)
                    weights = weights.get('state_dict',weights.get('model',weights))
                    weights = {k.removeprefix('module.'):v for k,v in weights.items()}
                    target = self.backbone.model if backbone_kind=='swin_t' else self.backbone
                    target.load_state_dict(weights,strict=True)
                if backbone_kind=='swin_t': self.backbone.model.head=nn.Identity()
                else: self.backbone.prediction=nn.Identity()
            elif cfg.get('paper_ablation',False):
                from .ablation import AblationBackbone
                self.backbone = AblationBackbone(cfg)
                input_dim = 512
            else:
                self.backbone = ResNet(BasicBlock, [2,2,2,2], hsg=cfg.get('hsg',True),
                                   graph_order=cfg.get('graph_order','tsg-lsg'),graph_conv=cfg.get('graph_conv','edge'))
                self.backbone.fc = nn.Identity()
                input_dim = 512
            path = cfg.get('resnet_weights')
            if initialize and path and backbone_kind not in ('swin_t','pvig_tiny'):
                weights = torch.load(path, map_location='cpu', weights_only=True)
                weights = {k:(v.unsqueeze(2) if ('conv' in k or 'downsample.0.weight' in k) else v)
                           for k,v in weights.items() if not k.startswith('fc.')}
                result = self.backbone.load_state_dict(weights, strict=False)
                if result.unexpected_keys or any(not k.startswith(('localG','temporalG','hsg','alpha','blocks.','gains.')) for k in result.missing_keys):
                    raise ValueError(f'Unexpected ResNet initialization keys: {result}')
            elif initialize and not cfg.get('allow_random_init',False) and backbone_kind not in ('swin_t','pvig_tiny'):
                raise ValueError('Provide local resnet_weights; random initialization is smoke-test only')
        else:
            self.backbone = None
            input_dim = cfg['feature_dim']
        self.input_dim = input_dim
        self.conv1d = TemporalConv(input_dim, hidden, conv_type=2, use_bn=True)
        self.temporal = BiLSTMLayer(input_size=hidden,hidden_size=hidden,num_layers=2,dropout=.3)
        self.classifier = NormLinear(hidden,len(vocab))
        self.kd = SeqKD(T=8)
        self.translation = None
        if cfg.get('task','cslr') == 'slt':
            path = cfg['mbart_path']
            lang = cfg.get('language','de_DE')
            self.tokenizer,self.translation = load_translation(path,lang)
            dim = self.translation.config.d_model
            self.mapping = nn.Sequential(nn.Linear(hidden,dim),nn.ReLU(),nn.Linear(dim,dim))
            if cfg.get('translation_checkpointing',True):
                self.translation.gradient_checkpointing_enable()
                self.translation.config.use_cache=False

    def features(self, video, lengths):
        # Save only checkpoint boundaries on CPU in the bounded mode. Segment
        # recomputation preserves RNG and isolates BatchNorm running buffers.
        active=video.is_cuda and self.training and torch.is_grad_enabled()
        if self.backbone is not None:
            self.backbone.activation_checkpoint=active and self.activation_offload=='cpu_checkpoint'
            from .graphs import EdgeConv
            for module in self.backbone.modules():
                if isinstance(module,EdgeConv):
                    module.activation_checkpoint=active and self.activation_offload=='cpu_checkpoint'
        if self.activation_offload in ('cpu','cpu_checkpoint') and active:
            with torch.autograd.graph.save_on_cpu(pin_memory=False):
                return self._features(video,lengths)
        return self._features(video,lengths)

    def _features(self, video, lengths):
        if self.backbone is not None:
            b,t,c,h,w = video.shape
            x = self.backbone(video.transpose(1,2)).reshape(b,t,self.input_dim).transpose(1,2)
        else:
            x = video.transpose(1,2)
        conv = self.conv1d(x,lengths.cpu())
        lengths = conv['feat_len'].long().cpu()
        if (lengths<=0).any():
            raise ValueError('Video too short for temporal convolution')
        temporal = self.temporal(conv['visual_feat'], lengths)['predictions']
        return {'conv':self.classifier(conv['visual_feat']), 'sequence':self.classifier(temporal),
                'features':temporal.transpose(0,1),'lengths':lengths}

    def translation_inputs(self, features, lengths):
        z = self.mapping(features)
        embedding = self.translation.model.shared.weight
        langid = self.tokenizer.lang_code_to_id[self.cfg.get('language','de_DE')]
        suffix = embedding[[self.tokenizer.eos_token_id,langid]].to(z.dtype)
        values, masks = [], []
        maximum = int(lengths.max())+2
        for feature,length in zip(z,lengths.tolist()):
            val = torch.cat([feature[:length],suffix])
            values.append(F.pad(val,(0,0,0,maximum-len(val))))
            masks.append(torch.arange(maximum,device=z.device)<length+2)
        return {'inputs_embeds':torch.stack(values)*math.sqrt(z.shape[-1]),
                'attention_mask':torch.stack(masks).long()}

    def forward(self, video, lengths, rows=None):
        out = self.features(video,lengths)
        if rows is None:
            return out
        field = self.cfg.get('target_field','gloss')
        labels, label_lengths, required = encode_targets(rows,self.vocab,field)
        if (required>out['lengths']).any():
            bad = [r['id'] for r,invalid in zip(rows,required>out['lengths']) if invalid]
            raise ValueError(f'CTC alignment impossible; labels NEVER truncated: {bad}')
        loss = out['sequence'].float().sum()*0
        for key,weight in [('conv',self.cfg.get('conv_ctc',1.)),('sequence',self.cfg.get('seq_ctc',1.))]:
            if weight:
                loss = loss+weight*F.ctc_loss(out[key].float().log_softmax(-1),labels,
                                             out['lengths'],label_lengths,blank=0,reduction='none',zero_infinity=False).mean()
        if self.cfg.get('distillation',25.):
            loss = loss+self.cfg.get('distillation',25.)*self.kd(out['conv'].float(),out['sequence'].detach().float(),use_blank=False)
        if self.translation is not None:
            tokens = self.tokenizer([r['text'] for r in rows],return_tensors='pt',padding=True)
            labels = tokens.input_ids.to(video.device)
            labels[labels==self.tokenizer.pad_token_id] = -100
            result = self.translation(**self.translation_inputs(out['features'],out['lengths']), labels=labels)
            ce = F.cross_entropy(result.logits.float().transpose(1,2),labels,ignore_index=-100,
                                 reduction='sum',label_smoothing=self.cfg.get('label_smoothing',.2))/len(rows)
            loss = loss+self.cfg.get('translation_weight',1.)*ce
        return loss

    @torch.no_grad()
    def generate(self, out):
        if self.translation is None:
            return ['']*len(out['lengths'])
        lang = self.cfg.get('language','de_DE')
        result = self.translation.generate(**self.translation_inputs(out['features'],out['lengths']),
            decoder_start_token_id=self.tokenizer.lang_code_to_id[lang],
            num_beams=self.cfg.get('text_beam',5),max_new_tokens=self.cfg.get('max_text_tokens',128),use_cache=True)
        return self.tokenizer.batch_decode(result,skip_special_tokens=True)
