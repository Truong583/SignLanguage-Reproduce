import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from PIL import Image

from scripts.doctor import check_data
from repro.data import SignDataset,collate,build_vocab,encode_targets
from repro.vendor.tconv import TemporalConv


def rgb_case(root):
    cfg={'task':'cslr','input_kind':'rgb','data_root':str(root),'target_field':'gloss'}
    for split,length in [('train',7),('dev',2),('test',16)]:
        folder=root/split; folder.mkdir()
        for i in range(length): Image.new('RGB',(16,16),(i,0,0)).save(folder/f'{i:04d}.png')
        row={'id':split+'_short','frames':split,'gloss':'a a b','text':''}
        manifest=root/(split+'.jsonl'); manifest.write_text(json.dumps(row)+'\n')
        cfg[split]=str(manifest)
    weights=root/'weights.pth'; weights.touch(); cfg['resnet_weights']=str(weights)
    return cfg


def test_preflight_matches_real_rgb_training_resampling_and_finite_ctc_backward(tmp_path,monkeypatch):
    import random
    cfg=rgb_case(tmp_path)
    monkeypatch.setattr(random.Random,'uniform',lambda *args:.8)
    summary=check_data(cfg,full=True)
    assert summary['train']['samples']==1
    assert summary['train']['ctc_length_adjusted_samples']==1
    assert summary['train']['short_sequence_examples'][0]['minimum_training_frames']==13
    dataset=SignDataset(cfg['train'],tmp_path,train=True)
    video,row=dataset[0]
    assert len(video)==13 and row['gloss']=='a a b'
    batch=collate([(video,row)])
    conv=TemporalConv(8,8,conv_type=2,use_bn=True).eval()
    features=torch.randn(1,8,int(batch['lengths'][0]),requires_grad=True)
    result=conv(features,batch['lengths'])
    vocab=build_vocab(dataset.rows,'gloss')
    labels,lengths,required=encode_targets([row],vocab,'gloss')
    assert result['feat_len'].tolist()==required.tolist()==[4]
    logits=torch.nn.Linear(8,len(vocab))(result['visual_feat'])
    loss=torch.nn.functional.ctc_loss(logits.log_softmax(-1),labels,result['feat_len'],lengths,zero_infinity=False)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(features.grad).all() and features.grad.abs().sum()>0


def test_eval_reports_short_sequences_without_changing_frames_or_references(tmp_path):
    cfg=rgb_case(tmp_path)
    before=Path(cfg['dev']).read_bytes()
    summary=check_data(cfg)
    assert summary['dev']['decode_length_limited_samples']==1
    assert summary['dev']['ctc_length_adjusted_samples']==0
    assert summary['test']['decode_length_limited_samples']==0
    video,row=SignDataset(cfg['dev'],tmp_path,train=False)[0]
    assert len(video)==2 and row['gloss']=='a a b'
    assert Path(cfg['dev']).read_bytes()==before


def test_doctor_cli_accepts_the_same_short_rgb_training_case(tmp_path):
    import yaml
    cfg=rgb_case(tmp_path)
    config=tmp_path/'config.yaml'; config.write_text(yaml.safe_dump(cfg))
    project=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,'scripts/doctor.py','--config',str(config),'--full'],cwd=project,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'Preflight passed' in result.stdout and '"ctc_length_adjusted_samples": 1' in result.stdout


def test_short_training_features_still_fail_before_training(tmp_path):
    cfg=rgb_case(tmp_path); cfg['input_kind']='features'; cfg['feature_dim']=8
    for split in ('train','dev','test'):
        np.save(tmp_path/(split+'.npy'),np.ones((7,8),dtype=np.float32))
        Path(cfg[split]).write_text(json.dumps({'id':split,'features':split+'.npy','gloss':'a a b'})+'\n')
    with pytest.raises(ValueError,match='Impossible CTC alignment.*split=train.*required=4'):
        check_data(cfg)


def test_preflight_still_rejects_empty_images_and_empty_labels(tmp_path):
    cfg=rgb_case(tmp_path)
    Path(cfg['train']).write_text(json.dumps({'id':'empty','frames':'train','gloss':''})+'\n')
    with pytest.raises(ValueError,match='Empty (CTC target|vocabulary)'):
        check_data(cfg)
    Path(cfg['train']).write_text(json.dumps({'id':'empty','frames':'train','gloss':'a'})+'\n')
    for path in (tmp_path/'train').iterdir(): path.unlink()
    with pytest.raises(ValueError,match='No frames'):
        check_data(cfg)
