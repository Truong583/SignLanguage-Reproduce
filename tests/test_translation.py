"""Exercise mBART wiring using random tiny weights and a tokenizer stub.

This verifies tensor/loss/generation integration, not real tokenization or BLEU.
"""
from types import SimpleNamespace
import torch
import transformers
from repro.data import collate
from repro.model import SignModel


class TokenizerStub:
    pad_token_id=1
    eos_token_id=2
    lang_code_to_id={'de_DE':4}
    @classmethod
    def from_pretrained(cls,*args,**kwargs): return cls()
    def __call__(self,texts,**kwargs):
        return SimpleNamespace(input_ids=torch.tensor([[5,6,2,4] for _ in texts]))
    def batch_decode(self,values,**kwargs):
        return [' '.join(str(int(t)) for t in row) for row in values]


def test_tiny_mbart_loss_generation_and_feature_gradients(tmp_path,monkeypatch):
    from transformers import MBartConfig,MBartForConditionalGeneration
    config=MBartConfig(vocab_size=16,d_model=16,encoder_layers=1,decoder_layers=1,
        encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=32,
        decoder_ffn_dim=32,max_position_embeddings=64,pad_token_id=1,bos_token_id=0,
        eos_token_id=2,decoder_start_token_id=4)
    MBartForConditionalGeneration(config).save_pretrained(tmp_path)
    monkeypatch.setattr('repro.model.load_translation',lambda path,lang:
                        (TokenizerStub(),MBartForConditionalGeneration.from_pretrained(path,local_files_only=True)))
    cfg={'task':'slt','input_kind':'features','feature_dim':8,'hidden_size':16,
         'mbart_path':str(tmp_path),'language':'de_DE','translation_checkpointing':False,
         'distillation':1.,'text_beam':2,'max_text_tokens':4}
    model=SignModel(cfg,['<blank>','<unk>','a','b'])
    batch=collate([(torch.randn(12,8),{'id':'one','gloss':'a b','text':'test sentence'})])
    loss=model(batch['video'],batch['lengths'],batch['rows'])
    assert torch.isfinite(loss)
    loss.backward()
    assert model.mapping[0].weight.grad.abs().sum()>0
    assert model.translation.model.shared.weight.grad.abs().sum()>0
    model.eval()
    with torch.no_grad():
        out=model.features(batch['video'],batch['lengths'])
        prepared=model.translation_inputs(out['features'],out['lengths'])
        assert prepared['attention_mask'].sum()==out['lengths'].sum()+2
        decoded=model.generate(out)
    assert len(decoded)==1
