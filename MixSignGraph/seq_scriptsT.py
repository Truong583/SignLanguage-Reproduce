import os, pdb, sys, copy, torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from einops import rearrange
from collections import defaultdict
from utils.misc import *
from utils.metrics import *
import loralib as lora

def seq_train(loader, model, optimizer, epoch_idx, recoder, accelerator):
    if accelerator.num_processes>1:
        print("num_class=", len(model.module.gloss_tokenizer), model.module.gloss_tokenizer.pad_id)
    model.train()
    optimizer.scheduler.step(epoch_idx)
    loss_value = []
    clr = [group['lr'] for group in optimizer.optimizer.param_groups]
    for batch_idx, data in enumerate(tqdm(loader)):
        vid, vid_lgt, gloss, text = data[0], data[1], data[2], data[3]
        optimizer.zero_grad()
        loss = model(vid, vid_lgt, gloss,  text)
        del vid, vid_lgt, gloss, text

        if np.isinf(loss.item()) or np.isnan(loss.item()):
            print('loss is nan', loss)
            print(str(data[1])+'  frames', str(data[2])+'  glosses')
        # print("debug==32", loss, data[-1][0]['name'])
        accelerator.backward(loss)
        optimizer.step()
        loss = accelerator.gather_for_metrics(loss).mean()
        loss_value.append(loss.item())
        if batch_idx % recoder.log_interval == 0 and accelerator.is_main_process:
            recoder.print_log('\tEpoch: {}, Batch({}/{}) done. Loss: {:.8f}  lr:{:.6f}'
                    .format(epoch_idx, batch_idx, len(loader), loss.item(), clr[0]))
        del loss, data
    optimizer.scheduler.step()
    if accelerator.is_main_process:
        recoder.print_log('\tMean training loss: {:.10f}.'.format(np.mean(loss_value)))
    return


def seq_eval(cfg, loader, model, mode, epoch, work_dir, recoder, accelerator=None):
    model.eval()
    results=defaultdict(dict)
    save=defaultdict(dict)
    for batch_idx, data in enumerate(tqdm(loader)):
        recoder.record_timer("device")
        vid,vid_lgt, gloss, text = data[0], data[1], data[2], data[3]
        info = [d['name'] for d in data[-1]]
        with torch.no_grad():
            if accelerator.num_processes>1:
                ret_dict = model.module.forward1(vid, vid_lgt, label=None, label_lgt=None)
            else:
                ret_dict = model.forward1(vid, vid_lgt, label=None, label_lgt=None)

            conv_sents, recognized_sents, text_pred=ret_dict["conv_sents"], ret_dict["recognized_sents"], ret_dict["text_pred"]
            ret_dict_conv = accelerator.gather_for_metrics(conv_sents)
            ret_dict_recg = accelerator.gather_for_metrics(recognized_sents)
            text_pred = accelerator.gather_for_metrics(text_pred)
            gloss_all = accelerator.gather_for_metrics(gloss)
            info_all = accelerator.gather_for_metrics(info)
            text_all = accelerator.gather_for_metrics(text) 
            torch.cuda.empty_cache()

            for inf, conv_sents, recognized_sents, gl,txtp, txtg in zip(info_all, ret_dict_conv, ret_dict_recg, gloss_all,text_pred,text_all):
                results[inf]['conv_sents'] = conv_sents[:-1]
                results[inf]['recognized_sents'] = recognized_sents[:-1]
                results[inf]['gloss'] = gl
                results[inf]['text_pred'] = txtp
                results[inf]['text'] = txtg
 
    gls_hyp = [' '.join(results[n]['conv_sents']) for n in results]
    gls_ref = [results[n]['gloss'] for n in results]
    wer_results_con = wer_list(hypotheses=gls_hyp, references=gls_ref)
    gls_hyp = [' '.join(results[n]['recognized_sents']) for n in results]
    wer_results = wer_list(hypotheses=gls_hyp, references=gls_ref)
    reg_per = wer_results if wer_results['wer'] < wer_results_con['wer'] else wer_results_con

    txt_hyp = [results[n]['text_pred'] for n in results]
    txt_ref = [results[n]['text'] for n in results]
    bleu_dict = bleu(hypotheses=txt_hyp, references=txt_ref, level=cfg.level)
    rouge_score = rouge(hypotheses=txt_hyp, references=txt_ref, level=cfg.level)
    tra_per = {**bleu_dict, **{'rouge': rouge_score}}
    if accelerator.is_main_process:
        recoder.print_log('\tEpoch: {} {} done. Conv wer: {:.4f}  ins:{:.4f}, del:{:.4f}'.format(
            epoch, mode, wer_results_con['wer'], wer_results_con['ins'], wer_results_con['del']),
            f"{work_dir}/{mode}.txt")
        recoder.print_log('\tEpoch: {} {} done. LSTM wer: {:.4f}  ins:{:.4f}, del:{:.4f}'.format(
            epoch, mode, wer_results['wer'], wer_results['ins'], wer_results['del']), f"{work_dir}/{mode}.txt")

        recoder.print_log(
            '\tEpoch: {} {} done. R: {:.4f}  b1:{:.4f}, b2:{:.4f} b3:{:.4f} b4:{:.4f}'.format(
                epoch, mode, tra_per['rouge'], tra_per['bleu1'], tra_per['bleu2'], tra_per['bleu3'], tra_per['bleu4']),
            f"{work_dir}/{mode}.txt")

    return {**reg_per, **tra_per}
