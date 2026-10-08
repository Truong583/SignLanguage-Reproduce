"""Explicit corpus WER and log-domain CTC prefix beam search (no LM)."""
import math
import numpy as np


def logadd(*values):
    maximum = max(values)
    if maximum == -math.inf:
        return maximum
    return maximum + math.log(sum(math.exp(v-maximum) for v in values))


def ctc_decode(log_probs, beam=10):
    if beam < 1:
        raise ValueError('beam must be >=1')
    p = np.asarray(log_probs)
    if beam == 1:
        result, previous = [], None
        for token in p.argmax(-1):
            if token != 0 and token != previous:
                result.append(int(token))
            previous = token
        return result
    hypotheses = {(): (0., -math.inf)}
    for frame in p:
        updated = {}
        def add(prefix, blank=-math.inf, nonblank=-math.inf):
            old = updated.get(prefix, (-math.inf, -math.inf))
            updated[prefix] = (logadd(old[0], blank), logadd(old[1], nonblank))
        # Full vocabulary expansion, no undocumented token pruning.
        for prefix, (pb, pn) in hypotheses.items():
            total = logadd(pb, pn)
            add(prefix, blank=total+float(frame[0]))
            for token in range(1, len(frame)):
                score = float(frame[token])
                if prefix and prefix[-1] == token:
                    add(prefix, nonblank=pn+score)
                    add(prefix+(token,), nonblank=pb+score)
                else:
                    add(prefix+(token,), nonblank=total+score)
        hypotheses = dict(sorted(updated.items(), key=lambda item: logadd(*item[1]), reverse=True)[:beam])
    return list(max(hypotheses, key=lambda h: logadd(*hypotheses[h])))


def edit_counts(reference, hypothesis):
    ref, hyp = reference.split(), hypothesis.split()
    # (cost, substitutions, deletions, insertions), deterministic tie breaking.
    previous = [(j, 0, 0, j) for j in range(len(hyp)+1)]
    for i, r in enumerate(ref, 1):
        current = [(i, 0, i, 0)]
        for j, h in enumerate(hyp, 1):
            if r == h:
                current.append(previous[j-1])
            else:
                c,s,d,ins = previous[j-1]
                diagonal = (c+1,s+1,d,ins)
                c,s,d,ins = previous[j]
                delete = (c+1,s,d+1,ins)
                c,s,d,ins = current[j-1]
                insert = (c+1,s,d,ins+1)
                current.append(min([diagonal, delete, insert], key=lambda x:x[0]))
        previous = current
    return (*previous[-1][1:], len(ref))


def corpus_wer(references, hypotheses):
    if len(references) != len(hypotheses):
        raise ValueError('Reference/hypothesis count mismatch')
    counts = [0,0,0,0]
    for ref,hyp in zip(references,hypotheses):
        counts = [a+b for a,b in zip(counts,edit_counts(ref,hyp))]
    s,d,i,n = counts
    if not n:
        raise ValueError('WER undefined for empty reference corpus')
    return {'wer':100*(s+d+i)/n,'sub':100*s/n,'del':100*d/n,'ins':100*i/n,'reference_words':n}


def translation_scores(refs, hyps, level='word'):
    # Same bundled BLEU/ROUGE implementations as upstream, imported independently.
    import importlib.util
    from pathlib import Path
    folder = Path(__file__).resolve().parents[1]/'MixSignGraph/utils'
    def load(name):
        spec = importlib.util.spec_from_file_location(name, folder/f'{name}.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    if level == 'char':
        refs = [' '.join(''.join(r.split())) for r in refs]
        hyps = [' '.join(''.join(h.split())) for h in hyps]
    sacre, rouge = load('sacrebleu'), load('Rouge')
    scores = sacre.raw_corpus_bleu(sys_stream=hyps,ref_streams=[refs]).scores
    value = rouge.rouge(hyps,refs)['rouge_l/f_score']
    return {**{f'bleu{i+1}':s for i,s in enumerate(scores)}, 'rouge':value*100}
