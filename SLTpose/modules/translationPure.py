import torch,sys,math, os
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from modules.tokenizer import *
from torch import nn, Tensor 
from peft import (get_peft_model, 
                  LoraConfig, 
                  TaskType, 
                  )

class CELoss(nn.Module):
    """
    Cross-Entropy Loss with optional label smoothing
    """
    def __init__(self, pad_index: int, smoothing: float = 0.0):
        super(CELoss, self).__init__()
        self.smoothing = smoothing
        self.pad_index = pad_index
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_index,
                                             label_smoothing=self.smoothing)
    def forward(self, logits, targets):
        logits = logits[..., :-1, :].contiguous()
        targets = targets[..., 1:].contiguous()
        loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss

def load_model(path, embeding_path=None, fc_path=None, padding_id=None): 
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path,
                                                 torch_dtype=torch.float16,
                                                 attention_dropout=0.3)
    if fc_path!=None:
        newlogit= torch.load(fc_path, map_location='cpu')
        model.config.vocab_size = newlogit.shape[0]
        model.lm_head = nn.Linear(model.config.hidden_size, model.config.vocab_size, bias=False)
        model.lm_head.weight = nn.Parameter(newlogit, requires_grad=True)
    if embeding_path!=None:
        newembeding = torch.load(embeding_path, map_location='cpu')
        model.model.embed_tokens = nn.Embedding(model.config.vocab_size,  model.config.hidden_size, padding_id)
        model.model.embed_tokens.weight = nn.Parameter(newembeding, requires_grad=True)
        # model.model.embed_tokens =None
    with open(os.path.join(path + '/config.json'), 'r') as f:
        config_json = json.load(f)
    print('Vocab size = ', tokenizer.vocab_size, config_json)
    loraconfig = LoraConfig(
        r=4,
        lora_alpha=32,
        lora_dropout=0.3,
        bias="none",
        use_rslora=True,
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj",'v_proj', ], #'lm_head', 'embed_tokens'
        modules_to_save=['lm_head', 'embed_tokens']
    )
    model = get_peft_model(model, loraconfig)
    model.print_trainable_parameters()
    return tokenizer, model, config_json

class SLTModel(nn.Module):
    def __init__(self, input_type="feature", cfg=None):
        super().__init__()
        self.input_type = input_type
        assert self.input_type in ['gloss','feature', 'feature+gloss']

        self.text_tokenizer = TextTokenizer(tokenizer_cfg=cfg["TextTokenizer"])
        self.beg_of_text_id = self.text_tokenizer.beg_of_text_id
        self.padding_id = self.text_tokenizer.padding_id
        self.end_of_text_id = self.text_tokenizer.end_of_text_id
        self.end_of_text_id2 = self.text_tokenizer.end_of_text_id2 

        _, self.model, _ = load_model(cfg['pretrained_model_name_or_path'], 
                                      embeding_path=cfg['embedding_file'],
                                      fc_path=cfg['logit_file'],
                                      padding_id= self.padding_id)
        self.input_dim = self.model.config.hidden_size
        self.input_embed_scale = cfg.get('input_embed_scale', math.sqrt(self.model.config.hidden_size))

        self.translation_loss_fun = CELoss(
            pad_index=-100,
            smoothing=cfg['label_smoothing'])
    
    def embed_tokens(self, input_ids):
        input_ids = torch.tensor(input_ids).to(self.model.device)
        return self.model.get_input_embeddings()(input_ids)

    def get_prompt_embeding(self,gloss, i):
        prompt_beg = self.embed_tokens(self.text_tokenizer.get_promte_ids("<video>", add_special_tokens=True))
        self.promt_len = len(prompt_beg)
        if gloss != None and self.input_type=='feature+gloss':
            if self.training and len(gloss[i][:-1]) > 1:
                gloss_str = " ".join(np.random.choice(gloss[i][:-1], int(len(gloss[i][:-1]) / 2) + 1, replace=False))
            else:
                gloss_str = " ".join(gloss[i][:-1])
            # gloss_str = " ".join(gloss[i][:-1])
            prompt_end = self.embed_tokens(self.text_tokenizer.get_promte_ids("</video> " + gloss_str, False))
        else:
            prompt_end = self.embed_tokens(self.text_tokenizer.get_promte_ids("</video>", False))

        return prompt_beg, prompt_end

    def prepare_feature_inputs(self, input_feature=None, input_lengths=None, text=None, text_lengths=None, gloss=None):
        b, t, d = input_feature.shape
        input_embedding=[]
        max_feature_len = 0
        visual_feature_len, text_feature_len = [], []
        text_ids, label_ids = None, None
        if text!=None:
            text_ids, label_ids = self.text_tokenizer.prepare_text_inputs(text)

        for i in range(b):
            prompt_beg, prompt_end = self.get_prompt_embeding(gloss, i)
            visual_input = torch.cat([prompt_beg, input_feature[i, :input_lengths[i], :], prompt_end], dim=0)
            visual_feature_len.append(visual_input.shape[0])
            if text==None:
                sentence_embed=torch.tensor([]).to(self.model.device)
            else: 
                sentence_embed = self.embed_tokens(text_ids[i])
            text_feature_len.append(len(sentence_embed))
            vis_sentence_embed = torch.cat((visual_input, sentence_embed),dim=0)
            max_feature_len = max(max_feature_len, vis_sentence_embed.shape[0]) 
            input_embedding.append(vis_sentence_embed)
        input_embedding_padding=[]
        attention_mask = torch.zeros([b, max_feature_len], dtype=torch.long, 
                                     device=input_feature.device)
        label = torch.ones([input_feature.shape[0], max_feature_len],
                            dtype=torch.long).fill_(-100)
        input_embedding_lengths=[]
        for i in range(b):
            padd_embeding = [self.embed_tokens([self.padding_id])[0]] * (max_feature_len - input_embedding[i].shape[0])
            input_embedding_lengths.append(input_embedding[i].shape[0])
            if len(padd_embeding) > 0:
                input_embedding_padding.append(torch.cat((input_embedding[i],torch.stack(padd_embeding))))
            else:
                input_embedding_padding.append(input_embedding[i]) 
            attention_mask[i, 0:input_embedding[i].shape[0]] = 1
            if text!=None: 
                label[i, visual_feature_len[i]:visual_feature_len[i]+len(label_ids[i])] = label_ids[i]
        return {
            "inputs_embeds": torch.stack(input_embedding_padding, dim=0),
            "attention_mask": attention_mask, 
            "visual_feature_len": torch.tensor(visual_feature_len, dtype=torch.long),
            "text_feature_len": torch.tensor(text_feature_len, dtype=torch.long),
            "input_embed_len": torch.tensor(input_embedding_lengths, dtype=torch.long),
            "labels": label.to(input_feature.device),
        }
     

    def prepare_feature_inputs_left_padding(self, input_feature, input_lengths, text=None, text_lengths=None, gloss=None):
        b, t, d = input_feature.shape
        input_embedding, visual_feature_len, text_feature_len=[], [], []
        text_ids, label_ids = None, None
        if text!=None:
            text_ids, label_ids = self.text_tokenizer.prepare_text_inputs(text)
        max_feature_len = 0
        for i in range(b):
            prompt_beg, prompt_end = self.get_prompt_embeding(gloss, i)
            visual_input = torch.cat([prompt_beg, input_feature[i, :input_lengths[i], :], prompt_end], dim=0)
            visual_feature_len.append(visual_input.shape[0])
            if text==None:
                sentence_embed=torch.tensor([]).to(self.model.device)
            else: 
                sentence_embed = self.embed_tokens(text_ids[i])
            text_feature_len.append(len(sentence_embed))
            vis_sentence_embed = torch.cat((visual_input, sentence_embed),dim=0)
            max_feature_len = max(max_feature_len, vis_sentence_embed.shape[0]) 
            input_embedding.append(vis_sentence_embed)
        input_embedding_padding=[]
        attention_mask = torch.zeros([b, max_feature_len], dtype=torch.long, 
                                     device=input_feature.device)
        label = torch.ones([input_feature.shape[0], max_feature_len],
                            dtype=torch.long).fill_(-100)
        input_embedding_lengths = []
        for i in range(b):
            padd_embeding = [self.embed_tokens([self.padding_id])[0]] * (max_feature_len - input_embedding[i].shape[0])
            input_embedding_lengths.append(input_embedding[i].shape[0])
            if len(padd_embeding) > 0:
                input_embedding_padding.append(torch.cat((torch.stack(padd_embeding), input_embedding[i])))
            else:
                input_embedding_padding.append(input_embedding[i])
            attention_mask[i, -input_embedding[i].shape[0]:] = 1
            if text!=None: 
                label[i, visual_feature_len[i]:visual_feature_len[i]+text_feature_len[i]:] = label_ids[i]
     
        return {
            "inputs_embeds": torch.stack(input_embedding_padding, dim=0),
            "attention_mask": attention_mask, 
            "visual_feature_len": torch.tensor(visual_feature_len, dtype=torch.long),
            "text_feature_len": torch.tensor(text_feature_len, dtype=torch.long),
            "input_embed_len": torch.tensor(input_embedding_lengths, dtype=torch.long),
            "labels": label.to(input_feature.device),
        }

    def forward(self,**kwargs):
        if 'feature' in self.input_type:
            input_feature = kwargs.pop('input_feature')
            input_lengths = kwargs.pop('input_lengths')
            text_str = kwargs.pop('text')
            gloss = kwargs.pop('gloss', None)
  
            new_kwargs = self.prepare_feature_inputs(input_feature, input_lengths, text_str, gloss=gloss)
            visual_feature_len = new_kwargs.pop('visual_feature_len', None)
            text_feature_len = new_kwargs.pop('text_feature_len', None)
            input_embed_len = new_kwargs.pop('input_embed_len', None)
            # lables = new_kwargs['labels']
            lables = new_kwargs.pop('labels')
            # keargs['']
            # attention_mask = new_kwargs.pop('attention_mask')
            kwargs = {**kwargs, **new_kwargs} 
 
        kwargs['output_hidden_states']=True
        with torch.cuda.amp.autocast():
            output_dict = self.model(**kwargs, return_dict=True)
        batch_loss_sum = self.translation_loss_fun(output_dict['logits'],lables)
        output_dict['total_loss'] = batch_loss_sum #output_dict['loss'] #
        output_dict['sequences'] = output_dict['logits'].argmax(-1)
        output_dict['sequences'] = [t[visual_feature_len[i]-1:visual_feature_len[i]+text_feature_len[i]-1]for i,t in enumerate(output_dict['sequences'])]
        for i in range(len(output_dict['sequences'])):
            if max(text_feature_len)-text_feature_len[i]>0:
                output_dict['sequences'][i] = torch.cat([output_dict['sequences'][i], torch.tensor([self.padding_id]).repeat(max(text_feature_len)-text_feature_len[i]).to(output_dict['sequences'][i].device)], dim=0)
        output_dict['sequences'] = torch.stack(output_dict['sequences'])


        return output_dict
    

    # def generate(self, gloss=None, num_beams=6, max_length=60, length_penalty=1, input_feature=None, input_lengths=None):
    #     final_output_dict={'decoded_sequences':[]}
    #     self.model.base_model.model.model.embed_tokens =self.embedding
    #     for i in range(len(input_feature)):
    #         nkwargs = self.prepare_feature_inputs(input_feature[0].unsqueeze(0), [input_lengths[i]], text=None)
    #         nkwargs.pop('visual_feature_len', None)
    #         nkwargs.pop('text_feature_len', None) 
    #         nkwargs.pop('labels', None)

    #         inputs_embeds, attention_mask = nkwargs['inputs_embeds'], nkwargs['attention_mask']
    #         assert inputs_embeds!=None and attention_mask!=None 
    #         with torch.cuda.amp.autocast(dtype=torch.float16):
    #             output_dict = self.model.generate(inputs_embeds=inputs_embeds, 
    #                                         attention_mask=None,
    #                                         num_beams=num_beams, 
    #                                         length_penalty=length_penalty, 
    #                                         top_p=1, 
    #                                         max_new_tokens=max_length, 
    #                                         pad_token_id=self.padding_id,
    #                                         eos_token_id=[self.end_of_text_id, self.end_of_text_id2],
    #                                         return_dict_in_generate=True, 
    #                                         output_hidden_states=False)
    #             print("1==", output_dict['sequences'])
    #             output_dict['decoded_sequences'] = self.text_tokenizer.batch_decode(output_dict['sequences'], skip_special_tokens=True)
    #             print("2==", output_dict['decoded_sequences'][0])
    #             final_output_dict['decoded_sequences'].append(output_dict['decoded_sequences'][0])
    #     return final_output_dict

    def generate(self, gloss=None, num_beams=6, max_length=60, length_penalty=1, input_feature=None, input_lengths=None):
        nkwargs = self.prepare_feature_inputs(input_feature , input_lengths, text=None, gloss=gloss)
        nkwargs.pop('visual_feature_len', None)
        nkwargs.pop('text_feature_len', None) 
        nkwargs.pop('labels', None)
        input_embed_len = nkwargs.pop('input_embed_len', None)

        inputs_embeds, attention_mask = nkwargs['inputs_embeds'], nkwargs['attention_mask']
        assert inputs_embeds!=None and attention_mask!=None 

        # causal_mask = torch.triu(torch.ones(inputs_embeds.shape[1], inputs_embeds.shape[1]), diagonal=1).to(torch.bool)
        # with torch.cuda.amp.autocast(dtype=torch.float16):
        with torch.cuda.amp.autocast():
            output_dict = self.model.generate(inputs_embeds=inputs_embeds,
                                        attention_mask=attention_mask,
                                        num_beams=20,  #
                                        length_penalty=length_penalty, 
                                        top_p=1,  #0.5,
                                        max_new_tokens=max_length,
                                        pad_token_id=self.padding_id,
                                        eos_token_id=[self.end_of_text_id, self.end_of_text_id2],
                                        return_dict_in_generate=True,
                                        repetition_penalty=1, 
                                        early_stopping=True,
                                        output_hidden_states=False)
        # print("1==", output_dict['sequences'])
        output_dict['decoded_sequences'] = self.text_tokenizer.batch_decode(output_dict['sequences'], skip_special_tokens=True)

        return output_dict
  

if __name__ == '__main__':
    with torch.device('cuda:0'):
        cfg ={'pretrained_model_name_or_path':'../../Dataprocessing/LLama3_8B',
            'label_smoothing':0.8,
            'TextTokenizer': {'pretrained_model_name_or_path': '../../Dataprocessing/LLama3_8B', 
                              'pruneids_file': '../../Dataprocessing/OpenASLLLama/map_ids.pkl'},
            'embedding_file': '../../Dataprocessing/OpenASLLLama/embedding.pt',
            'logit_file': '../../Dataprocessing/OpenASLLLama/logit.pt'}
        model =SLTModel(input_type='feature', cfg=cfg).to('cuda:0') 

        input_feature =torch.randn([2, 10, 4096]).to('cuda:0')
        input_lengths = [10, 1]
        text=['i am your facther', 'how about your mom after the party?']

        input_arg={"input_feature": input_feature,
                'input_lengths':input_lengths,
                'text':text}
                
        output = model(**input_arg)
        
        # print(output.keys())

        
        # input_arg={"input_feature": input_feature,
        #         'input_lengths':input_lengths}
                
        # output = model.generate(**input_arg)
        # print(output['decoded_sequences'])