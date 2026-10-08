import torch,sys,math
import torch.nn as nn
from transformers import MBartForConditionalGeneration, MBartTokenizer, MBartConfig, MBartModel
from modules.tokenizer import *
from utils.misc import freeze_params
from torch import nn, Tensor
from torch.autograd import Variable

class XentLoss(nn.Module):
    """
    Cross-Entropy Loss with optional label smoothing
    """

    def __init__(self, pad_index: int, smoothing: float = 0.0):
        super(XentLoss, self).__init__()
        self.smoothing = smoothing
        self.pad_index = pad_index
        if self.smoothing <= 0.0:
            # standard xent loss
            self.criterion = nn.NLLLoss(ignore_index=self.pad_index, reduction="sum")
        else:
            # custom label-smoothed loss, computed with KL divergence loss
            self.criterion = nn.KLDivLoss(reduction="sum")

    def _smooth_targets(self, targets, vocab_size):
        """
        Smooth target distribution. All non-reference words get uniform
        probability mass according to "smoothing".
        :param targets: target indices, batch*seq_len
        :param vocab_size: size of the output vocabulary
        :return: smoothed target distributions, batch*seq_len x vocab_size
        """
        # batch*seq_len x vocab_size
        smooth_dist = targets.new_zeros((targets.size(0), vocab_size)).float()
        # fill distribution uniformly with smoothing
        smooth_dist.fill_(self.smoothing / (vocab_size - 2))
        # assign true label the probability of 1-smoothing ("confidence")
        smooth_dist.scatter_(1, targets.unsqueeze(1).data, 1.0 - self.smoothing)
        # give padding probability of 0 everywhere
        smooth_dist[:, self.pad_index] = 0
        # masking out padding area (sum of probabilities for padding area = 0)
        padding_positions = torch.nonzero(targets.data == self.pad_index)
        # pylint: disable=len-as-condition
        if len(padding_positions) > 0:
            smooth_dist.index_fill_(0, padding_positions.squeeze(), 0.0)
        return Variable(smooth_dist, requires_grad=False)

    # pylint: disable=arguments-differ
    def forward(self, log_probs, targets):
        """
        Compute the cross-entropy between logits and targets.
        If label smoothing is used, target distributions are not one-hot, but
        "1-smoothing" for the correct target token and the rest of the
        probability mass is uniformly spread across the other tokens.
        :param log_probs: log probabilities as predicted by model
        :param targets: target indices
        :return:
        """
        if self.smoothing > 0:
            targets = self._smooth_targets(
                targets=targets.contiguous().view(-1), vocab_size=log_probs.size(-1)
            )
            # targets: distributions with batch*seq_len x vocab_size
            assert (
                log_probs.contiguous().view(-1, log_probs.size(-1)).shape
                == targets.shape
            )
        else:
            # targets: indices with batch*seq_len
            targets = targets.contiguous().view(-1)
        loss = self.criterion(
            log_probs.contiguous().view(-1, log_probs.size(-1)), targets
        )
        return loss

def load_model(path):
    model = MBartForConditionalGeneration.from_pretrained(path, attention_dropout= 0.1, dropout= 0.3)
    return model


class SLTModel(nn.Module):
    def __init__(self, input_type="feature", cfg=None):
        super().__init__()
        self.input_type = input_type
        assert self.input_type in ['gloss','feature', 'feature+gloss']
        self.model = load_model(cfg['pretrained_model_name_or_path'])

        self.input_dim = self.model.config.d_model
        self.input_embed_scale = cfg.get('input_embed_scale', math.sqrt(self.model.config.d_model))
        #
        # self.gloss_tokenizer = GlossTokenizer_G2T(tokenizer_cfg=cfg["GlossTokenizer"])
        # self.gloss_embedding = self.build_gloss_embedding(**cfg['GlossEmbedding'])
        self.text_tokenizer = TextTokenizer(tokenizer_cfg=cfg["TextTokenizer"])

        self.translation_loss_fun = XentLoss(
            pad_index=self.text_tokenizer.pad_index,
            smoothing=cfg['label_smoothing'])


    def prepare_text_inputs(self, input_str):
        input_ids = self.text_tokenizer(input_str)
        return input_ids["labels"], input_ids["decoder_input_ids"]

    def prepare_feature_inputs(self, input_feature, input_lengths, gloss=None, gloss_lengths=None):
        suffix_embedding = [self.model.model.shared.weight[self.text_tokenizer.eos_index,:]]
        src_lang_id = self.text_tokenizer.lang_index # self.text_tokenizer.pruneids[self.tokenizer.convert_tokens_to_ids(self.tokenizer.tgt_lang)]

        # print("debug11==", src_lang_id, self.text_tokenizer.tokenizer.decode([self.text_tokenizer.pruneids_reverse[src_lang_id]]))
        # assert src_lang_id == 31
        src_lang_code_embedding = self.model.model.shared.weight[src_lang_id,:]
        suffix_embedding.append(src_lang_code_embedding)
        suffix_len = len(suffix_embedding)
        suffix_embedding = torch.stack(suffix_embedding, dim=0)
        max_length = torch.max(input_lengths)+suffix_len
        if 'gloss' in self.input_type:
            gloss = self.text_tokenizer.getids(gloss)
            for ii, in_len in enumerate(input_lengths):
                max_length = max(max_length, in_len + len(gloss[ii])+suffix_len)


        inputs_embeds = []
        attention_mask = torch.zeros([input_feature.shape[0], max_length], dtype=torch.long, device=input_feature.device)

        for ii, feature in enumerate(input_feature):
            valid_len = input_lengths[ii]
            valid_feature = feature[:valid_len,:] #t,D

            if 'gloss' in self.input_type:
                gloss_lengths = len(gloss[ii]) #input_lengths[ii]+
                gloss_emdb = torch.stack([self.model.model.shared.weight[glossids] for glossids in gloss[ii]]) #g_l D
                # print("debug1==", valid_feature.shape, gloss_emdb.shape)
                # valid_feature = torch.cat([valid_feature, gloss_emdb.detach()], dim=0) # t+g_l, D
                # valid_len = valid_len + gloss_lengths
                # print("debug111=",  valid_feature.shape)
                # print("debug22==", valid_feature.shape)
                valid_feature = torch.cat( [gloss_emdb, feature[:valid_len - gloss_lengths, :]], dim=0)
            if suffix_embedding != None:
                feature_w_suffix = torch.cat([valid_feature, suffix_embedding], dim=0) # t+2, D
            else:
                feature_w_suffix = valid_feature
            if feature_w_suffix.shape[0]<max_length:
                pad_len = max_length-feature_w_suffix.shape[0]
                padding = torch.zeros([pad_len, feature_w_suffix.shape[1]],
                    dtype=feature_w_suffix.dtype, device=feature_w_suffix.device)
                padded_feature_w_suffix = torch.cat([feature_w_suffix, padding], dim=0) #t+2+pl,D
                inputs_embeds.append(padded_feature_w_suffix)
            else:
                inputs_embeds.append(feature_w_suffix)
            attention_mask[ii, :valid_len+suffix_len] = 1
        transformer_inputs = {
            'inputs_embeds': torch.stack(inputs_embeds, dim=0)*self.input_embed_scale, #B,T,D
            'attention_mask': attention_mask#attention_mask
        }
        return transformer_inputs

    def forward(self,**kwargs):
        if self.input_type=='gloss':
            kwargs.pop("name", None)
            gloss_str = kwargs.pop('gloss')
            text_str = kwargs.pop('text')
            input_feature = kwargs.pop('input_feature')
            input_lengths = kwargs.pop('input_lengths')
            kwargs.pop('gloss_ids', None)
            kwargs.pop('gloss_lengths', None)
            kwargs['input_ids'], kwargs['attention_mask'] = self.prepare_gloss_inputs(gloss_str)
            kwargs["labels"], kwargs["decoder_input_ids"] = self.prepare_text_inputs(text_str)
            kwargs['input_ids'] = kwargs['input_ids'].to(input_feature.device)
            # print("debug1=",gloss_str,  kwargs['inputs_ids'])
            # print("Debug==", kwargs['attention_mask'])
            kwargs['attention_mask'] = kwargs['attention_mask'].to(input_feature.device)
            kwargs["labels"] = kwargs["labels"].to(input_feature.device)
            kwargs["decoder_input_ids"] = kwargs["decoder_input_ids"].to(input_feature.device)


        elif 'feature' in self.input_type:
            input_feature = kwargs.pop('input_feature')
            input_lengths = kwargs.pop('input_lengths')
            text_str = kwargs.pop('text')
            #quick fix
            kwargs.pop('gloss_ids', None)
            kwargs.pop('gloss_lengths', None)
            gloss = kwargs.pop('gloss', None)

            new_kwargs = self.prepare_feature_inputs(input_feature, input_lengths, gloss=gloss)
            kwargs["labels"], kwargs["decoder_input_ids"] = self.prepare_text_inputs(text_str)
            kwargs["labels"] = kwargs["labels"].to(input_feature.device)
            kwargs["decoder_input_ids"] = kwargs["decoder_input_ids"].to(input_feature.device)
            kwargs = {**kwargs, **new_kwargs}

        # print("debug190==", kwargs["labels"].size(), input_feature.size())
        kwargs['output_hidden_states']=True
        output_dict = self.model(**kwargs, return_dict=True)
        # print(output_dict.keys())#, loss, logits, past_key_values, encoder_last_hidden_state
        # print(output_dict["encoder_last_hidden_state"].shape) # b, len, 1024
        # print(output_dict["loss"])  # b, len, 1024
        log_prob = torch.nn.functional.log_softmax(output_dict['logits'], dim=-1)  # B, T, L
        batch_loss_sum = self.translation_loss_fun(log_probs=log_prob,targets=kwargs['labels'])
        output_dict['total_loss'] = batch_loss_sum/log_prob.shape[0]
        # print(batch_loss_sum, output_dict['total_loss'])
        # output_dict['transformer_inputs'] = kwargs #for later use (decoding)

        output_dict['decoded_sequences'] = output_dict['logits']

        # print(output_dict.keys()) #odict_keys(['loss', 'logits', 'decoder_hidden_states', 'encoder_last_hidden_state', 'encoder_hidden_states', 'total_loss', 'transformer_inputs'])
        # print(output_dict['decoder_hidden_states'][-1].shape)torch.Size([2, 31, 1024])

        return output_dict

    def prepare_gloss_inputs(self, gloss):

        suffix_embedding = [self.model.model.shared.weight[self.text_tokenizer.eos_index, :]]
        src_lang_id = self.text_tokenizer.lang_index  # self.text_tokenizer.pruneids[self.tokenizer.convert_tokens_to_ids(self.tokenizer.tgt_lang)]

        # print("debug11==", src_lang_id, self.text_tokenizer.tokenizer.decode([self.text_tokenizer.pruneids_reverse[src_lang_id]]))
        # assert src_lang_id == 31
        src_lang_code_embedding = self.model.model.shared.weight[src_lang_id, :]
        suffix_embedding.append(src_lang_code_embedding)
        suffix_len = len(suffix_embedding)
        suffix_embedding = torch.stack(suffix_embedding, dim=0)

        gloss,attention_mask = self.text_tokenizer.getidsforgloss(gloss)
        return gloss.to(self.model.device), torch.tensor(attention_mask).to(self.model.device)
        # max_length = max([len(g_t) + suffix_len for g_t in gloss ])
        # inputs_embeds = []
        # attention_mask = torch.zeros([gloss.shape[0], max_length], dtype=torch.long,
        #                              device=gloss.device)
        # for ii in range(len(gloss)):
        #     max_length = max(max_length, len(gloss[ii]) + suffix_len)
        #     valid_len = len(gloss[ii]) + suffix_len
        #     gloss_emdb = torch.stack([self.model.model.shared.weight[glossids] for glossids in gloss[ii]])  # g_l D
        #     if suffix_embedding != None:
        #         gloss_emdb = torch.cat([gloss_emdb, suffix_embedding], dim=0)  # t+2, D
        #     inputs_embeds.append(gloss_emdb)
        #     attention_mask[ii, :valid_len] = 1
        # # torch.stack(inputs_embeds, dim=0) * self.input_embed_scale
        # inputs_embeds = torch.stack(inputs_embeds, dim=0).to(self.model.device) #* self.input_embed_scale
        # attention_mask = attention_mask.to(self.model.device)
        # return gloss, attention_mask
        # transformer_inputs = {
        #     'inputs_embeds': torch.stack(inputs_embeds, dim=0) * self.input_embed_scale,  # B,T,D
        #     'attention_mask': attention_mask  # attention_mask
        # }
        # # print("transformer_inputs==", transformer_inputs['inputs_embeds'].shape)
        # return transformer_inputs

    #phoenix 14t 100
    def generate(self, gloss=None, num_beams=6, max_length=60, length_penalty=1,
                 input_feature=None, input_lengths=None):
        if self.input_type=='gloss':
            inputs_ids, attention_mask = self.prepare_gloss_inputs(gloss)
            assert attention_mask != None
            batch_size = attention_mask.shape[0]
            decoder_input_ids = torch.ones([batch_size, 1], dtype=torch.long,
                                           device=attention_mask.device) * self.text_tokenizer.sos_index
            assert inputs_ids != None and attention_mask != None
            output_dict = self.model.generate(
                input_ids=inputs_ids, attention_mask=attention_mask,  # same with forward
                decoder_input_ids=decoder_input_ids,
                num_beams=num_beams, length_penalty=length_penalty, max_length=max_length,
                return_dict_in_generate=True, output_hidden_states=True)
            output_dict['decoded_sequences'] = self.text_tokenizer.batch_decode(output_dict['sequences'])
            # print("dddd==", output_dict.keys())
            return output_dict

        else:
            nkwargs = self.prepare_feature_inputs(input_feature, input_lengths, gloss)
            inputs_embeds, attention_mask = nkwargs['inputs_embeds'], nkwargs['attention_mask']
        assert attention_mask!=None
        batch_size = attention_mask.shape[0]
        decoder_input_ids = torch.ones([batch_size,1],dtype=torch.long, device=attention_mask.device)*self.text_tokenizer.sos_index
        assert inputs_embeds!=None and attention_mask!=None
        output_dict = self.model.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, #same with forward
            decoder_input_ids=decoder_input_ids,
            num_beams=num_beams, length_penalty=length_penalty, max_length=max_length,
            return_dict_in_generate=True, output_hidden_states=True)
        output_dict['decoded_sequences'] = self.text_tokenizer.batch_decode(output_dict['sequences'])
        # print("dddd==", output_dict.keys())
        return output_dict


