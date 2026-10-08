"""Experiment inventory transcribed from paper tables and Figure 6 (not measured scores)."""
from itertools import permutations, product


def experiments():
    rows=[]
    def add(ident, source, changes=None, dev=None, test=None, note=None):
        rows.append({'id':ident,'source':source,'changes':changes or {},
                     'paper_dev_wer':dev,'paper_test_wer':test,'assumption':note})
    add('main','Table 5 / Table 1 full',dev=16.7,test=19.0)
    add('modules_none','Table 1',{'module_mask':[False]*6},22.3,22.2)
    names=['lsg1','tsg1','hsg1','lsg2','tsg2','hsg2']
    dev=[19.2,19.6,19.9,19.3,19.5,19.4]; test=[21.,21.5,21.6,20.8,21.2,21.5]
    for i,name in enumerate(names):
        add('only_'+name,'Table 1',{'module_mask':[j==i for j in range(6)]},dev[i],test[i])
    for ids,d,t in [([0,3],18.7,20.6),([1,4],18.6,20.2),([2,5],17.8,19.6),
                    ([0,1,2],17.1,19.7),([3,4,5],17.4,19.6)]:
        add('modules_'+'_'.join(names[i] for i in ids),'Table 1',{'module_mask':[j in ids for j in range(6)]},d,t)
    for order,d,t in [('hsg-lsg-tsg',16.8,19.6),('hsg-tsg-lsg',16.9,19.5),
                      ('lsg-hsg-tsg',17.2,19.8),('lsg-tsg-hsg',17.1,19.0),('tsg-hsg-lsg',16.9,19.3)]:
        add('order_'+order,'Table 13b',{'module_order':order},d,t)
    scores=[(16.7,19.0),(18.1,20.1),(19.3,20.1),(19.9,21.8),
            (19.0,20.2),(19.1,21.1),(20.2,20.6),(20.7,21.5)]
    for (l,t,h),(d,v) in zip(product(('dense','sparse'),('sparse','dense'),('fixed','dynamic')),scores):
        add(f'types_{l}_{t}_{h}','Table 13a',{'lsg_type':l,'tsg_type':t,'hsg_type':h,'graph_scales':[16]},d,v,
            'Paper says one LSG/TSG/HSG but does not identify stage; reconstructed at patch16. Dynamic HSG uses KNN over projected high+low nodes.')
    for key,base,values in [('local_k',[3,4],range(2,10)),('temporal_k',[49,49],[7,21,35,49,63,77,91])]:
        for index in range(2):
            for value in values:
                if value==base[index]: continue
                choice=base.copy(); choice[index]=value
                add(f'{key}_{index+1}_{value}','Figure 6',{key:choice},note='Only one K varies; other three fixed. Figure values are not digitized into numeric targets.')
    for distance,d in [('cosine',17.4),('chebyshev',17.3)]:
        add('distance_'+distance,'Table 14b',{'distance':distance},d,
            note='LSG retains released normalized features/relative-position bias and projection widths; distance changes consistently in LSG/TSG. Exact author distance normalization is not released.')
    for conv,d in [('gatv2',17.0),('sage',17.7),('gcn',17.7)]:
        add('conv_'+conv,'Table 14c',{'graph_conv':conv},d,
            note='GATv2 single head, SAGE mean aggregation. Original hidden widths/head counts and LSG adapters are not released.')
    for backbone,d in [('swin_t',45.4),('pvig_tiny',35.4),('sa',39.2)]:
        add('backbone_'+backbone,'Table 14a',{'backbone':backbone},d,
            note='Swin-T/PyViG-Tiny ImageNet feature trunks; PyViG size not specified in paper. SA: 8 heads, spatial/crossscale self-attention and adjacent-frame-pair attention.')
    for scale,d in [(8,17.4),(16,17.1),(32,17.5)]:
        add(f'patch_{scale}','Table 14d',{'graph_scales':[scale]},d,
            note='One graph triplet at the corresponding ResNet feature stride; stems and training recipe retained.')
    for scales,d in [([8,16,32],17.1),([4,8,16,32],17.8)]:
        add('stages_'+'_'.join(map(str,scales)),'Table 14e',{'graph_scales':scales},d,
            note='Additional early triplets use first-stage K; patch4 HSG bridges conv1 to layer1 across maxpool.')
    for rate,d in [(.15,17.7),(.30,17.5)]:
        add(f'dropedge_{int(rate*100)}','Table 14f',{'drop_edge':rate},d,
            note='Independent edge Bernoulli dropout at each training forward in LSG/HSG; LSG masks messages after its edge BatchNorm, before aggregation. No dropout during eval. Author mask schedule is not released.')
    add('multisigngraph','Table 3 qualitative reference',{'module_mask':[True,True,False,True,True,False]},
        note='Reconstructed MultiSignGraph reference removes both HSG modules; exact baseline checkpoint is not released. No numeric paper WER is assigned to this qualitative reference.')
    assert len({r['id'] for r in rows})==len(rows)==68
    di={
      'main':(4.9,2.1,4.9,2.9),'modules_none':(8.4,2.5,8.1,2.7),
      'only_lsg1':(5.6,2.2,4.8,2.3),'only_tsg1':(5.6,2.1,5.1,2.5),'only_hsg1':(6.7,2.1,6.2,3.1),
      'only_lsg2':(5.3,2.1,5.8,2.0),'only_tsg2':(6.6,1.7,5.4,2.4),'only_hsg2':(6.2,2.9,5.9,3.6),
      'modules_lsg1_lsg2':(5.1,2.3,5.2,1.7),'modules_tsg1_tsg2':(4.3,1.8,5.5,1.7),'modules_hsg1_hsg2':(6.1,2.3,5.3,2.9),
      'modules_lsg1_tsg1_hsg1':(5.3,2.0,5.0,3.0),'modules_lsg2_tsg2_hsg2':(5.7,2.1,5.1,2.9),
      'order_hsg-lsg-tsg':(5.1,2.0,5.0,3.2),'order_hsg-tsg-lsg':(5.1,2.1,4.9,3.2),
      'order_lsg-hsg-tsg':(6.3,2.0,4.8,2.7),'order_lsg-tsg-hsg':(5.4,2.0,5.0,2.7),'order_tsg-hsg-lsg':(5.5,1.8,5.0,3.2),
      'types_dense_sparse_fixed':(4.9,2.1,4.9,2.9),'types_dense_sparse_dynamic':(5.1,2.2,5.1,2.1),
      'types_dense_dense_fixed':(4.9,2.9,5.2,2.6),'types_dense_dense_dynamic':(5.6,3.5,5.5,2.6),
      'types_sparse_sparse_fixed':(5.3,2.1,5.4,2.1),'types_sparse_sparse_dynamic':(5.0,1.9,5.8,2.1),
      'types_sparse_dense_fixed':(5.3,2.6,5.5,2.9),'types_sparse_dense_dynamic':(6.0,2.1,6.0,2.5),
      'distance_cosine':(5.5,2.0),'distance_chebyshev':(5.1,3.2),
      'conv_gatv2':(5.1,2.0),'conv_sage':(5.9,1.9),'conv_gcn':(5.5,2.0),
      'backbone_swin_t':(16.3,1.3),'backbone_pvig_tiny':(12.1,1.3),'backbone_sa':(15.3,0.9),
      'patch_8':(5.4,3.0),'patch_16':(5.3,2.0),'patch_32':(5.6,2.4),
      'stages_8_16_32':(5.2,2.0),'stages_4_8_16_32':(5.4,3.1),
      'dropedge_15':(6.1,1.4),'dropedge_30':(6.0,2.0)}
    for row in rows:
        values=di.get(row['id'],())
        for i,key in enumerate(('paper_dev_del','paper_dev_ins','paper_test_del','paper_test_ins')):
            row[key]=values[i] if i<len(values) else None
    return rows


BASE={'paper_ablation':True,'backbone':'resnet18','module_mask':[True]*6,
      'module_order':'tsg-lsg-hsg','graph_scales':[16,32],'local_k':[3,4],
      'temporal_k':[49,49],'distance':'euclidean','lsg_type':'dense',
      'tsg_type':'sparse','hsg_type':'fixed','drop_edge':0.,'graph_conv':'edge'}
