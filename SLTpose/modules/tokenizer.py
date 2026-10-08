from json import decoder

import numpy as np
import torch, pickle, json
from collections import defaultdict
from transformers import MBartTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, ignore_index: int = -100):
    """
    Shift input ids one token to the right, and wrap the last non pad token (the <LID> token) Note that MBart does not
    have a single `decoder_start_token_id` in contrast to other Bart-like models.
    """
    prev_output_tokens = input_ids.clone()

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)
    index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    for ii, ind in enumerate(index_of_eos.squeeze(-1)):
        input_ids[ii, ind:] = ignore_index
    decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
    # prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
    # change
    prev_output_tokens[:, 1:] = input_ids[:, :-1].clone()
    prev_output_tokens[:, 0] = decoder_start_tokens
    return prev_output_tokens


class BaseTokenizer(object):
    def __init__(self, tokenizer_cfg):
        self.tokenizer_cfg = tokenizer_cfg

    def __call__(self, input_str):
        pass


class TextTokenizer(BaseTokenizer):
    def __init__(self, tokenizer_cfg):
        super().__init__(tokenizer_cfg)  
        self.language= tokenizer_cfg['language']
        self.text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_cfg['pretrained_model_name_or_path'])
        # self.pad_token ='<|eot_id|>'
        self.pad_token='<|finetune_right_pad_id|>'
        self.pruneids_file = tokenizer_cfg['pruneids_file']
        with open(self.pruneids_file, 'rb') as f:
            self.pruneids = pickle.load(f)
     
        self.pruneids_reverse = {i2: i1 for i1, i2 in self.pruneids.items()}

        self.padding_id= self.pruneids[self.text_tokenizer.convert_tokens_to_ids(self.pad_token)]
        self.beg_of_text_id = self.pruneids[self.text_tokenizer.convert_tokens_to_ids("<|begin_of_text|>")]
        self.end_of_text_id = self.pruneids[self.text_tokenizer.convert_tokens_to_ids("<|end_of_text|>")]
        self.end_of_text_id2 = self.pruneids[self.text_tokenizer.convert_tokens_to_ids("<|eot_id|>")]
 

    def get_promte_ids(self, promote, add_special_tokens=False):
        input_ids = self.text_tokenizer(promote,  add_special_tokens=add_special_tokens)['input_ids']
        input_ids = self.prune_single(input_ids, promote)
        return input_ids

    def prepare_text_inputs(self, input_str):
        input_list=[]
        label_list=[] 
        for stri in input_str:
            input_ids = self.text_tokenizer(stri, add_special_tokens=False)['input_ids']
            input_list.append(input_ids +[self.text_tokenizer.convert_tokens_to_ids("<|end_of_text|>")])
            label_list.append(input_ids +[self.text_tokenizer.convert_tokens_to_ids("<|end_of_text|>")])
        input_list = self.prune(input_list)
        label_list = self.prune(label_list) 
        return input_list, label_list 



    def prune_single(self, input_ids, promote=None):
        pruned_input_ids = []  
        for id_ in input_ids:
            if not id_ in self.pruneids:
                print("debug tokenizer lin80", id_, self.text_tokenizer.decode(id_), promote)
                continue
                # new_id = self.pruneids[self.text_tokenizer.convert_tokens_to_ids('<unk>')]
            else:
                new_id = self.pruneids[id_]
            pruned_input_ids.append(new_id)
        return torch.tensor(pruned_input_ids, dtype=torch.long)

    def prune(self, input_ids):
        pruned_input_ids = []
        for single_seq in input_ids:
            pruned_single_seq = self.prune_single(single_seq)
            pruned_input_ids.append(pruned_single_seq) 
        return pruned_input_ids 

    def prune_reverse(self, pruned_input_ids):
        batch_size, max_len = pruned_input_ids.shape
        input_ids = pruned_input_ids 
        for b in range(batch_size):
            for i in range(max_len):
                id_ = input_ids[b, i].item()
                if not id_ in self.pruneids_reverse:
                    print("debug tokenizer lin82", id_, self.text_tokenizer.decode(id_))
                    new_id = self.text_tokenizer.eos_token_id
                else:
                    new_id = self.pruneids_reverse[id_]
                input_ids[b, i] = new_id
        return input_ids
    
    def batch_decode(self, sequences, skip_special_tokens=True): 
        sequences_ = self.prune_reverse(sequences) 
        decoded_sequences = self.text_tokenizer.batch_decode(sequences_, skip_special_tokens=skip_special_tokens)
        for i in range(len(decoded_sequences)):
            if self.language=='de':
                if len(decoded_sequences[i]) >= 2 and decoded_sequences[i][-1] == '.' and decoded_sequences[i][-2] != ' ':
                    decoded_sequences[i] = decoded_sequences[i][:-1]+ ' .'
            elif self.language=='openasl':
                decoded_sequences[i] = decoded_sequences[i].replace("'m", " 'm").replace("'s", " 's")
        return decoded_sequences

    def __len__(self):
        return  len(self.pruneids)


class BaseGlossTokenizer(BaseTokenizer):
    def __init__(self, tokenizer_cfg):
        super().__init__(tokenizer_cfg)
        with open(tokenizer_cfg['gloss2id_file'], 'rb') as f:
            self.gloss2id = pickle.load(f)  #
        self.gloss2id = defaultdict(lambda: self.gloss2id['<unk>'], self.gloss2id)
        self.id2gloss = {}
        for gls, id_ in self.gloss2id.items():
            self.id2gloss[id_] = gls
        self.lower_case = True  # tokenizer_cfg.get('lower_case',True)

    def convert_tokens_to_ids(self, tokens):
        if type(tokens) == list or type(tokens) == np.array:
            return [self.convert_tokens_to_ids(t) for t in tokens]
        else:
            return self.gloss2id[tokens]

    def convert_ids_to_tokens(self, ids):
        if type(ids) == list:
            return [self.convert_ids_to_tokens(i) for i in ids]
        else:
            return self.id2gloss[ids]

    def __len__(self):
        return len(self.id2gloss)


class GlossTokenizer_S2G(BaseGlossTokenizer):
    def __init__(self, tokenizer_cfg):
        super().__init__(tokenizer_cfg) 
        # self.pad_token = '<|eot_id|>'
        self.pad_token ='<pad>'
        self.pad_id = self.convert_tokens_to_ids(self.pad_token)

    def __call__(self, batch_gls_seq, ifpad=False):
        # print("debug",batch_gls_seq)
        max_length = max([len(gls_seq.split(" ")) for gls_seq in batch_gls_seq])
        gls_lengths, batch_gls_ids = [], []
        for ii, gls_seq in enumerate(batch_gls_seq):
            # gls_ids = [self.gloss2id[gls.lower() if self.lower_case else gls] for gls in gls_seq.split()]
            gls_ids = []
            for gls in gls_seq.split():
                if gls.lower() in self.gloss2id.keys():
                    gls_ids.append(self.gloss2id[gls.lower()])
                else:
                    print("unknow token", gls, "|", gls_seq)
            gls_lengths.append(len(gls_ids))
            if ifpad:
                gls_ids = gls_ids + (max_length - len(gls_ids)) * [self.pad_id]
                batch_gls_ids.append(gls_ids)
            else:
                batch_gls_ids.extend(gls_ids)
        gls_lengths = torch.tensor(gls_lengths)
        batch_gls_ids = torch.LongTensor(batch_gls_ids)
        return {'gls_lengths': gls_lengths, 'gloss_labels': batch_gls_ids}

    def covert_ids_to_seq(self, batch_ids):
        if type(batch_ids) == torch.tensor:
            batch_ids = batch_ids.cpu().numpy().tolist()
        batch_gls_seq = []
        for ids in batch_ids:
            gls_seq = self.convert_tokens_to_ids(ids)
            batch_gls_seq.append(" ".join(gls_seq))
        return batch_gls_seq


class GlossTokenizer_G2T(BaseGlossTokenizer):
    def __init__(self, tokenizer_cfg):
        super().__init__(tokenizer_cfg)
        self.src_lang = tokenizer_cfg['src_lang']

    def __call__(self, batch_gls_seq):
        # batch
        max_length = max([len(gls_seq.split()) for gls_seq in batch_gls_seq]) + 2  # include </s> <lang>
        batch_gls_ids = []
        attention_mask = torch.zeros([len(batch_gls_seq), max_length], dtype=torch.long)
        for ii, gls_seq in enumerate(batch_gls_seq):
            gls_ids = [self.gloss2id[gls.lower() if self.lower_case else gls] for gls in gls_seq.split()]
            # add </s> <lang> + padding
            gls_ids = gls_ids + [self.gloss2id['</s>'], self.gloss2id[self.src_lang]]
            attention_mask[ii, :len(gls_ids)] = 1
            gls_ids = gls_ids + (max_length - len(gls_ids)) * [self.gloss2id['<pad>']]
            batch_gls_ids.append(gls_ids)
        input_ids = torch.tensor(batch_gls_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}


if __name__ == '__main__':
    cfg={}
    cfg['pretrained_model_name_or_path'] = '../../Dataprocessing/LLama3_8B'
    cfg["pruneids_file"] = '../../Dataprocessing/OpenASLLLama/map_ids.pkl'
    tokenizers = TextTokenizer(cfg)
    input,label = tokenizers.prepare_text_inputs(
        ['he was taken to the hospital but police said he died a few hours later at the hospital from his injuries',
         "also tonight the house of representatives are planning to vote on a resolution tonight that will condemn trump's tweet"])
    print(label)

    print(tokenizers.batch_decode(torch.stack([label[0]])))

'''

[   0,  146,    8, 1250,   21,   42, 2391,   98,  240,   11,  346,  336, 32,  712,  603,  270,   36,   42, 2391,  196,  143, 1222, 3040]),
[   0, 1766, 4575,   42,  204,  172, 2041,  241, 1955,   21, 2007,   64, 32, 1375, 4575,   88,  297, 8577,  304,  781,  726])], [

[ 146,    8, 1250,   21,   42, 2391,   98,  240,   11,  346,  336,   32,
         712,  603,  270,   36,   42, 2391,  196,  143, 1222, 3040, 3036]),

[1766, 4575,   42,  204,  172, 2041,  241, 1955,   21, 2007,   64,   32, 1375, 4575,   88,  297, 8577,  304,  781,  726, 3036])])
'''