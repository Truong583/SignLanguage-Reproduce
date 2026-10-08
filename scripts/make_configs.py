"""Generate explicit independent configs; no hidden YAML inheritance."""
from pathlib import Path
import copy
import yaml

root=Path(__file__).resolve().parents[1]
folder=root/'configs_repro'
base=yaml.safe_load((folder/'phoenix14t_cslr.yaml').read_text())
def save(name,cfg):
    (folder/f'{name}.yaml').write_text('# RECONSTRUCTION: see docs/AUDIT.md\n'+yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')

for dataset in ['phoenix14','csl_daily']:
    cfg=copy.deepcopy(base)
    cfg.update(data_root=f'data/{dataset}',output=f'runs/{dataset}_cslr_seed0')
    for split in ['train','dev','test']: cfg[split]=f'data/manifests/{dataset}/{split}.jsonl'
    save(f'{dataset}_cslr',cfg)

for dataset in ['phoenix14t','csl_daily','how2sign','openasl']:
    cfg=copy.deepcopy(base)
    cfg.update(data_root=f'data/{dataset}',target_field='pseudo_gloss')
    if dataset in ['how2sign','openasl']:
        cfg.update(input_kind='features',feature_dim=1024)
        cfg.pop('resnet_weights')
    for split in ['train','dev','test']: cfg[split]=f'data/manifests/{dataset}_tctc/{split}.jsonl'
    cfg['output']=f'runs/{dataset}_tctc_seed0'
    save(f'{dataset}_tctc',cfg)
    cfg.update(task='slt',mbart_path='assets/mbart-large-cc25',language={'phoenix14t':'de_DE','csl_daily':'zh_CN'}.get(dataset,'en_XX'),
               translation_lr=1e-5,mapping_lr=1e-3,translation_weight=1.,label_smoothing=.2,
               translation_checkpointing=True,text_beam=5,max_text_tokens=128,
               text_level='char' if dataset=='csl_daily' else 'word',selection_metric='rouge',
               output=f'runs/{dataset}_gfslt_seed0')
    save(f'{dataset}_gfslt',cfg)
    if dataset in ['phoenix14t','csl_daily']:
        cfg['target_field']='gloss'
        for split in ['train','dev','test']: cfg[split]=f'data/manifests/{dataset}/{split}.jsonl'
        cfg['output']=f'runs/{dataset}_slt_seed0'
        save(f'{dataset}_slt',cfg)
