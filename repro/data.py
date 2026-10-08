"""JSONL manifests, train-only vocabulary, deterministic per-sample augmentation."""
import json
from contextlib import contextmanager
import zipfile
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def read_manifest(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    ids = [r['id'] for r in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError(f'Empty manifest or duplicate IDs: {path}')
    return rows


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Path leaves dataset root: {relative}')
    return path


def frame_paths(root, row):
    folder = safe_path(root, row['frames'])
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})


@contextmanager
def frame_source(root,row):
    if 'clip_id' in row or 'archive' in row:
        if 'clip_id' in row:
            from .drive_data import clip_path
            path=clip_path(row)
        else: path=safe_path(root,row['archive'])
        with zipfile.ZipFile(path) as archive:
            names=sorted(n for n in archive.namelist() if Path(n).suffix.lower() in {'.png','.jpg','.jpeg'})
            if len(names)!=len(set(names)): raise ValueError('Duplicate frame names in video ZIP')
            if row.get('frame_count',len(names))!=len(names): raise ValueError('Video frame count mismatch')
            yield names,lambda name: archive.open(name)
    else:
        yield frame_paths(root,row),lambda path: path


class SignDataset(Dataset):
    def __init__(self, manifest, root, train=False, seed=0, input_kind='rgb',target_field='gloss'):
        self.rows = read_manifest(manifest)
        self.root, self.train, self.seed, self.epoch = root, train, seed, 0
        self.input_kind = input_kind
        self.target_field = target_field

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        rng = random.Random(self.seed + self.epoch * 1000003 + index)
        if self.input_kind == 'features':
            arr = np.load(safe_path(self.root, row['features']), allow_pickle=False)
            if arr.ndim != 2 or len(arr) < 1 or not np.isfinite(arr).all():
                raise ValueError(f'Invalid [T,D] feature array: {row["id"]}')
            video = torch.from_numpy(arr.astype(np.float32))
        else:
            with frame_source(self.root,row) as (paths,open_frame):
                if not paths:
                    raise ValueError(f'No frames: {row["id"]}')
                if self.train:
                    length = max(1, int(len(paths) * rng.uniform(.8, 1.2)))
                    tokens=row.get(self.target_field,'').split()
                    required=len(tokens)+sum(a==b for a,b in zip(tokens,tokens[1:]))
                    # Keep temporal augmentation valid for CTC, including repeats.
                    # Base alignment is checked by doctor; labels are never truncated.
                    length=max(length,4*required-3)
                    if length < len(paths):
                        indices = sorted(rng.sample(range(len(paths)), length))
                    else:
                        indices = sorted(list(range(len(paths))) + rng.choices(range(len(paths)), k=length-len(paths)))
                    paths = [paths[i] for i in indices]
                left, top = (rng.randint(0, 32), rng.randint(0, 32)) if self.train else (16, 16)
                flip = self.train and rng.random() < .5
                frames = []
                for path in paths:
                    with Image.open(open_frame(path)) as image:
                        image = image.convert('RGB').resize((256, 256), Image.Resampling.LANCZOS)
                        image = image.crop((left, top, left+224, top+224))
                        if flip:
                            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                        frames.append(np.asarray(image).copy())
                video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().div_(255)
                mean = video.new_tensor([.485, .456, .406])[None, :, None, None]
                std = video.new_tensor([.229, .224, .225])[None, :, None, None]
                video = (video - mean) / std
        return video, row


def collate(batch):
    # K5 P2 K5 P2: 6 boundary replicas at each end; output ceil(T/4).
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    lengths = torch.tensor([math.ceil(len(v)/4)*4+12 for v, _ in batch], dtype=torch.long)
    maximum = int(lengths.max())
    values = []
    for value, _ in batch:
        left = value[:1].expand(6, *value.shape[1:])
        right = value[-1:].expand(maximum-len(value)-6, *value.shape[1:])
        values.append(torch.cat([left, value, right]))
    return {'video':torch.stack(values), 'lengths':lengths, 'rows':[r for _, r in batch]}


def build_vocab(rows, field):
    tokens = sorted({token for row in rows for token in row[field].split()})
    if not tokens or '<blank>' in tokens or '<unk>' in tokens:
        raise ValueError('Empty vocabulary or reserved tokens in labels')
    return ['<blank>', '<unk>'] + tokens


def encode_targets(rows, vocab, field):
    lookup = {s:i for i, s in enumerate(vocab)}
    sequences = [[lookup.get(t, 1) for t in row[field].split()] for row in rows]
    if any(not seq for seq in sequences):
        raise ValueError('Empty CTC target; fix annotation instead of skipping')
    labels = torch.tensor([t for seq in sequences for t in seq], dtype=torch.long)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    required = torch.tensor([len(seq)+sum(a==b for a,b in zip(seq,seq[1:])) for seq in sequences])
    return labels, lengths, required
