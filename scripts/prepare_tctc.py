"""Reconstructed punctuation removal + lemmatization + word tokenization.

The author's exact NLP models/settings were not released. Record this choice.
"""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifests',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--spacy-model',required=True,help='Local model directory, not an automatic download')
    a=p.parse_args()
    import spacy
    nlp=spacy.load(a.spacy_model,disable=['ner','parser'])
    if 'lemmatizer' not in nlp.pipe_names: raise ValueError('Model must include lemmatizer')
    a.output.mkdir(parents=True,exist_ok=True)
    for split in ['train','dev','test']:
        target=a.output/f'{split}.jsonl'
        if target.exists(): raise FileExistsError(target)
        rows=[json.loads(s) for s in (a.manifests/f'{split}.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
        for row,doc in zip(rows,nlp.pipe([r['text'] for r in rows])):
            row['pseudo_gloss']=' '.join((t.lemma_ or t.text).lower() for t in doc if not t.is_punct and not t.is_space)
            if not row['pseudo_gloss']: raise ValueError(f'Empty pseudo label: {row["id"]}')
        target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    (a.output/'tctc_metadata.json').write_text(json.dumps({'spacy_version':spacy.__version__,'model':nlp.meta,
        'status':'reconstructed preprocessing; author settings unavailable'},indent=2),encoding='utf-8')

if __name__=='__main__': main()
