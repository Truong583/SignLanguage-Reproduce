import os
import pdb
import time
import torch 
import numpy as np
from itertools import groupby
import torch.nn.functional as F
class Decode(object):
    def __init__(self, gloss_dict, num_classes, search_mode, blank_id=0, beam_width=10):
        self.gloss_dict = gloss_dict
        self.num_classes = num_classes
        self.search_mode = search_mode
        self.blank_id = blank_id
        self.beam_width= beam_width 
       
    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2) 
        return self.MaxDecode(nn_output, vid_lgt) 

    def MaxDecode(self, nn_output, vid_lgt):
        index_list = torch.argmax(nn_output, axis=2)
        batchsize, lgt = index_list.shape
        ret_list = []
        for batch_idx in range(batchsize):
            group_result = [x[0] for x in groupby(index_list[batch_idx][:vid_lgt[batch_idx]])]
            filtered = [*filter(lambda x: x != self.blank_id, group_result)]
            if len(filtered) > 0:
                max_result = torch.stack(filtered)
                max_result = [x[0] for x in groupby(max_result)]
            else:
                max_result = filtered
            ret_list.append([self.gloss_dict.convert_ids_to_tokens(int(gloss_id)) for idx, gloss_id in
                             enumerate(max_result)]+["<pad>"])
            # ret_list.append([(self.gloss_dict.convert_ids_to_tokens[int(gloss_id)], idx) for idx, gloss_id in
            #                  enumerate(max_result)])
        return ret_list


    # def tf_BeamSearch(self,nn_output, vid_lgt, probs=False):
    #     if not probs:
    #         nn_output = nn_output.softmax(-1).permute(1,0,2).cpu().detach().numpy()
    #     vid_lgt = vid_lgt.cpu()

    #     # with tf.device('/device:cpu:0'):
    #     #T b v
    #     # print(nn_output.shape)
    #     # print("debug==", self.gloss_dict.pad_id)
    #     # nn_output = np.concatenate(
    #     #     (nn_output[:, :, 1:], nn_output[:, :, 0, None]),
    #     #     axis=-1,
    #     # )
    #     part_before = nn_output[:,:,:self.blank_id]

    #     part_after = nn_output[:,:,self.blank_id+1:]

    #     part_blank = nn_output[:,:,self.blank_id:self.blank_id+1]
    #     nn_output = np.concatenate(
    #         (part_before, part_after, part_blank),
    #         axis=-1,
    #     )
    #     results, _ = tf.nn.ctc_beam_search_decoder(nn_output, vid_lgt, 4, top_paths=4)

    #     results = results[0]
    #     tmp_gloss_sequences = [[] for i in range(vid_lgt.shape[0])]
    #     for (value_idx, dense_idx) in enumerate(results.indices):
    #         temt = results.values[value_idx].numpy()
    #         if temt>self.blank_id:
    #             temt+=1
    #         if results.values[value_idx].numpy()!=self.blank_id or True:
    #             tmp_gloss_sequences[dense_idx[0]].append(
    #                 temt #results.values[value_idx].numpy()
    #             )
    #     ret_list =[]
    #     decoded_gloss_sequences = []
    #     for seq_idx in range(0, len(tmp_gloss_sequences)):
    #         decoded_gloss_sequences.append(
    #             [x[0] for x in groupby(tmp_gloss_sequences[seq_idx])]
    #         )
    #     for first_result in decoded_gloss_sequences:
    #         ret_list.append([self.gloss_dict.convert_ids_to_tokens(int(gloss_id)) for idx, gloss_id in
    #                          enumerate(first_result)] + ["<pad>"])
    #     return ret_list
    #     # print(tmp_gloss_sequences)
    #     # print("debug1==", ret_list)