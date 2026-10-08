"""Export reviewable paper inventory and configuration files; no training or metrics."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import yaml
from repro.suite import BASE,experiments

output=ROOT/'configs_repro/phoenix14t_suite'; output.mkdir(exist_ok=True)
template=yaml.safe_load((ROOT/'configs_repro/phoenix14t_cslr.yaml').read_text())
rows=experiments()
(output/'inventory.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for row in rows:
    cfg={**template,**BASE,**row['changes'],'suite_experiment':row,
         'output':f'runs/phoenix14t_cslr_suite_seed0/{row["id"]}'}
    if cfg['backbone'] in ('swin_t','pvig_tiny'):
        cfg['backbone_weights']='assets/'+('swin_t-704ceda3.pth' if cfg['backbone']=='swin_t' else 'pvig_ti_78.5.pth.tar')
    (output/f'{row["id"]}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding='utf-8')
print(f'Exported {len(rows)} paper-scoped configurations.')
