"""Prepare manifests from official PHOENIX CSV or explicit portable JSONL.

No downloading, changing labels, dropping samples, or pickle loading here.
"""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
from itertools import groupby

ROOT=Path(__file__).resolve().parents[1]

def cleaners():
    # Extract only the two audited pure functions, avoiding upstream imports.
    source=ROOT/'MixSignGraph/utils/video_augmentation.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    selected=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in
              ['clean_phoenix_2014','clean_phoenix_2014_trans']]
    namespace={'re':re,'groupby':groupby}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),namespace)
    return namespace


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['phoenix14t','phoenix14','generic'],required=True)
    parser.add_argument('--annotations',type=Path,required=True,help='Directory with train/dev/test CSV or JSONL')
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--csv-template',default='PHOENIX-2014-T.{split}.corpus.csv')
    parser.add_argument('--frames-template',help='Relative to data root, e.g. features/fullFrame-210x260px/{split}/{name}')
    parser.add_argument('--delimiter',default='|')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    clean=cleaners()
    args.output.mkdir(parents=True,exist_ok=True)
    stats={}
    all_ids=set()
    prepared={}
    for split in ['train','dev','test']:
        if args.dataset=='generic':
            path=args.annotations/f'{split}.jsonl'
            rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        else:
            path=args.annotations/args.csv_template.format(split=split)
            with path.open(encoding='utf-8-sig',newline='') as f: source=list(csv.DictReader(f,delimiter=args.delimiter))
            rows=[]
            template=args.frames_template or ('features/fullFrame-210x260px/{split}/{name}' if args.dataset=='phoenix14t' else 'features/fullFrame-210x260px/{split}/{name}/1')
            normalizer=clean['clean_phoenix_2014_trans' if args.dataset=='phoenix14t' else 'clean_phoenix_2014']
            for row in source:
                name=row['name']
                gloss=row.get('orth') or row.get('annotation') or row.get('gloss')
                if not gloss: raise ValueError(f'Missing gloss in {path}: {name}')
                rows.append({'id':name,'frames':template.format(split=split,name=name),
                             'gloss':normalizer(gloss).lower(),
                             'text':(row.get('translation') or row.get('text') or '').strip().lower()})
        if not rows: raise ValueError(f'Empty split: {split}')
        for row in rows:
            if row['id'] in all_ids: raise ValueError(f'Duplicate ID across splits: {row["id"]}')
            all_ids.add(row['id'])
            key='features' if 'features' in row else 'frames'
            item=(args.data_root/row[key]).resolve()
            if not item.is_relative_to(args.data_root.resolve()): raise ValueError(f'Unsafe path: {item}')
            if not item.exists(): raise FileNotFoundError(item)
            if key=='frames' and not any(p.suffix.lower() in {'.jpg','.jpeg','.png'} for p in item.iterdir()):
                raise ValueError(f'No images: {item}')
        target=args.output/f'{split}.jsonl'
        if target.exists() and not args.overwrite: raise FileExistsError(target)
        prepared[target]=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows)
        stats[split]={'samples':len(rows),'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    for target,text in prepared.items(): target.write_text(text,encoding='utf-8')
    (args.output/'preparation.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    print(json.dumps(stats,indent=2))


if __name__=='__main__': main()
