"""Figure-5-style edge overlays and Table-3-style qualitative output from real runs.

No hand-selected background/important-edge labels; plotted edge subsets are deterministic.
"""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def render(output,data_root=None,manifests=None):
    import torch,yaml
    from PIL import Image,ImageDraw
    from repro.data import SignDataset,collate
    from repro.model import SignModel
    from repro.graphs import hierarchical_edges
    output=Path(output); main=output/'main'
    predictions=json.loads((main/'test_predictions.json').read_text())
    target='darunter nebel lang in-kommend daneben sonne berg oben dann durchgehend sonne'
    match=next((r for r in predictions if ' '.join(r['reference'].split())==target),predictions[0])
    qualitative={'sample_id':match['id'],'groundtruth':match['reference'],'mixsigngraph':match['sequence'],
      'selection':'Paper Table3 full reference matched' if ' '.join(match['reference'].split())==target else 'First test sample: paper sample not found',
      'status':'predictions from measured reconstruction, not copied paper predictions'}
    baseline=output/'multisigngraph/test_predictions.json'
    if baseline.exists():
        rows=json.loads(baseline.read_text()); reference=next((r for r in rows if r['id']==match['id']),None)
        if reference: qualitative['multisigngraph']=reference['sequence']
    (output/'qualitative.json').write_text(json.dumps(qualitative,ensure_ascii=False,indent=2),encoding='utf-8')
    # Drawing only has to run once; qualitative updates when the reference completes.
    if all((output/f'graph_stage{s}.png').exists() for s in (16,32)): return
    cfg=yaml.safe_load((main/'suite-config.yaml').read_text())
    if data_root is not None: cfg['data_root']=str(data_root)
    if manifests is not None: cfg['test']=str(Path(manifests)/'test.jsonl')
    state=torch.load(main/'best.pt',map_location='cpu',weights_only=False)
    model=SignModel(cfg,state['vocab'],initialize=False)
    model.load_state_dict(state['model'],strict=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device).eval()
    dataset=SignDataset(cfg['test'],cfg['data_root'])
    index=next(i for i,r in enumerate(dataset.rows) if r['id']==match['id'])
    batch=collate([dataset[index]])
    images=[]
    for index in (6,7):
        frame=batch['video'][0,index].permute(1,2,0)
        frame=(frame*frame.new_tensor([.229,.224,.225])+frame.new_tensor([.485,.456,.406])).clamp(0,1)
        images.append(Image.fromarray((frame.numpy()*255).round().astype('uint8')))
    records={}; handles=[]
    for scale in (16,32):
        local=model.backbone.blocks[f's{scale}_lsg'].graph.graph_conv.dilated_knn_graph
        def hook(module,args,result,scale=scale): records[scale]=result.detach().cpu()
        handles.append(local.register_forward_hook(hook))
        model.backbone.blocks[f's{scale}_tsg'].capture_edges=True
    with torch.no_grad(),torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
        model.backbone(batch['video'].to(device).transpose(1,2))
    for handle in handles: handle.remove()
    full={}
    for scale in (16,32):
        side=224//scale; n=side*side
        local=records[scale][:,6].reshape(2,-1)
        temporal=model.backbone.blocks[f's{scale}_tsg'].recorded_edges
        mask=(temporal[0]//n==6)&(temporal[1]//n==7)
        temporal=temporal[:,mask]-torch.tensor([[6*n],[7*n]])
        hierarchical=hierarchical_edges(side*2,side*2,side,side)
        full[str(scale)]={'lsg':local.tolist(),'tsg':temporal.tolist(),'hsg':hierarchical.tolist()}
        canvas=Image.new('RGB',(448,3*248),'white'); draw=ImageDraw.Draw(canvas)
        for row,name in enumerate(('LSG','TSG','HSG')):
            y=row*248+24
            canvas.paste(images[0],(0,y)); canvas.paste(images[1] if name=='TSG' else images[0],(224,y))
            draw.text((8,row*248+5),f'{name} patch{scale}: sample {match["id"]}',fill='black')
            edges={'LSG':local,'TSG':temporal,'HSG':hierarchical}[name]
            step=max(1,edges.shape[1]//150)
            for src,dst in edges[:,::step].T.tolist():
                if name=='HSG':
                    if src>=4*n: continue
                    dst-=4*n
                    point1=((src%(side*2)+.5)*(scale/2),(src//(side*2)+.5)*(scale/2)+y)
                    point2=((dst%side+.5)*scale+224,(dst//side+.5)*scale+y)
                else:
                    point1=((src%side+.5)*scale,(src//side+.5)*scale+y)
                    point2=((dst%side+.5)*scale+(224 if name=='TSG' else 0),(dst//side+.5)*scale+y)
                draw.line([point1,point2],fill={'LSG':'yellow','TSG':'cyan','HSG':'lime'}[name],width=1)
        canvas.save(output/f'graph_stage{scale}.png')
    (output/'graph_edges.json').write_text(json.dumps({'sample':match['id'],'frames':'first two real frames after collation padding',
      'note':'Actual reconstructed-model edges; deterministic subset plotted, no manual edge importance labels.', 'edges':full}))
