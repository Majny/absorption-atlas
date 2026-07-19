#!/usr/bin/env python3
"""
Feasibility gate for the mechanism experiment: can we MODULATE Gemma-2-2b's is-capitalized ICL task
accuracy via prompt format / few-shot balance? If accuracy spans a range within one model+property,
the "behavioral absorption is gated by task accuracy" demonstration is cheap and clean (hold model &
property fixed, vary only the prompt -> vary accuracy -> watch behavioral absorption track it while
representational absorption stays fixed).

Reports, per prompt variant: P(positive token -> positive answer), P(negative -> negative), and a
balanced accuracy. If some variants reach high accuracy and others low, the modulation works.
"""
import os
import random
from pathlib import Path

if "HF_TOKEN" not in os.environ:
    for c in (Path(__file__).parent / ".env", Path.cwd() / ".env"):
        if c.exists():
            for line in c.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip()
            break

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sae_spelling.experiments.common import DEFAULT_DEVICE, load_gemma2_model  # noqa: E402
from sae_spelling.vocab import get_alpha_tokens  # noqa: E402

random.seed(0)
CAP = lambda w: w.strip()[:1].isupper()  # noqa: E731

# (name, query_template, few_shot_line(word)->str, pos_answer, neg_answer, balanced_examples)
VARIANTS = [
    ("orig_random", "{word} starts with a capital letter:",
     lambda w, a: f"{w} starts with a capital letter:{a}", " True", " False", False),
    ("orig_balanced", "{word} starts with a capital letter:",
     lambda w, a: f"{w} starts with a capital letter:{a}", " True", " False", True),
    ("isupper_yesno", "Is {word} uppercase-initial? Answer:",
     lambda w, a: f"Is {w} uppercase-initial? Answer:{a}", " Yes", " No", True),
    ("case_word", "The first letter of {word} is:",
     lambda w, a: f"The first letter of {w} is:{a}", " uppercase", " lowercase", True),
]


def main():
    model = load_gemma2_model()
    tok = model.tokenizer
    vocab = get_alpha_tokens(tok)
    pos_pool = [w for w in vocab if CAP(w)]
    neg_pool = [w for w in vocab if not CAP(w)]
    print(f"device={DEFAULT_DEVICE}; vocab={len(vocab)} pos={len(pos_pool)}", flush=True)

    def acc_for(variant):
        name, qt, fsl, pa, na, balanced = variant
        pa_tok = tok.encode(pa, add_special_tokens=False)[-1]
        na_tok = tok.encode(na, add_special_tokens=False)[-1]

        def build(query_word):
            if balanced:
                ex = random.sample(pos_pool, 5) + random.sample(neg_pool, 5)
            else:
                ex = random.sample(vocab, 10)
            random.shuffle(ex)
            lines = [fsl(w, pa if CAP(w) else na) for w in ex if w != query_word]
            return "\n".join(lines) + "\n" + qt.format(word=query_word)

        def margins(words):
            out = []
            for i in range(0, len(words), 16):
                prompts = [build(w) for w in words[i:i + 16]]
                lens = {model.to_tokens(p).shape[1] for p in prompts}
                # process same-length groups
                by = {}
                for p in prompts:
                    by.setdefault(model.to_tokens(p).shape[1], []).append(p)
                for L, ps in by.items():
                    with torch.inference_mode():
                        lg = model(model.to_tokens(ps))
                    out.extend((lg[:, -1, pa_tok] - lg[:, -1, na_tok]).float().cpu().tolist())
            return out

        sp = random.sample(pos_pool, 64)
        sn = random.sample(neg_pool, 64)
        mp, mn = margins(sp), margins(sn)
        p_pos = float(np.mean([m > 0 for m in mp]))     # capitalized -> positive answer
        p_neg = float(np.mean([m < 0 for m in mn]))     # non-cap -> negative answer
        bal_acc = 0.5 * (p_pos + p_neg)
        print(f"{name:16} P(cap->pos)={p_pos:.2f} P(noncap->neg)={p_neg:.2f} bal_acc={bal_acc:.2f}", flush=True)
        return bal_acc

    for v in VARIANTS:
        acc_for(v)
    print("VARIANTS_DONE", flush=True)


if __name__ == "__main__":
    main()
