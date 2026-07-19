#!/usr/bin/env python3
"""
Task-validity check for the binary absorption properties. The absorption metric only means something
if the model ACTUALLY performs the True/False ICL task — otherwise IG attributions are ~0 for a
boring reason (no causal signal), not because absorption is spelling-specific. Probe AUROC~1.0 shows
the property is REPRESENTED, not that it is CAUSALLY used to answer.

For each property we compute the answer metric (True_logit - False_logit) on positive vs negative
tokens. If positives score >> negatives (high separation AUC), the model does the task and the
absorption result is meaningful. If pos≈neg, the metric is inapplicable.
"""
import os
import random
from collections import defaultdict
from pathlib import Path

if "HF_TOKEN" not in os.environ:
    for cand in (Path(__file__).parent / ".env", Path.cwd() / ".env"):
        if cand.exists():
            for line in cand.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip()
            break

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from run_binary import PROPERTIES, bin_formatter  # noqa: E402
from sae_spelling.experiments.common import DEFAULT_DEVICE, load_gemma2_model  # noqa: E402
from sae_spelling.prompting import create_icl_prompt  # noqa: E402
from sae_spelling.vocab import get_alpha_tokens  # noqa: E402

random.seed(0)
N = 96


def main():
    model = load_gemma2_model()
    tok = model.tokenizer
    pos_tok = tok.encode(" True", add_special_tokens=False)[-1]
    neg_tok = tok.encode(" False", add_special_tokens=False)[-1]
    vocab = get_alpha_tokens(tok)

    def metric_for(words, formatter, template):
        bases = [create_icl_prompt(w, examples=vocab, base_template=template,
                                   answer_formatter=formatter, max_icl_examples=10).base for w in words]
        by_len = defaultdict(list)
        for b in bases:
            by_len[model.to_tokens(b).shape[1]].append(b)
        vals = []
        with torch.no_grad():
            for _L, ps in by_len.items():
                for i in range(0, len(ps), 16):
                    toks = model.to_tokens(ps[i:i + 16])
                    logits = model(toks)
                    vals.extend((logits[:, -1, pos_tok] - logits[:, -1, neg_tok]).float().cpu().tolist())
        return vals

    print(f"device={DEFAULT_DEVICE}", flush=True)
    for name, cfg in PROPERTIES.items():
        pred, template = cfg["predicate"], cfg["template"]
        pos = [w for w in vocab if pred(w)]
        neg = [w for w in vocab if not pred(w)]
        sp = random.sample(pos, min(N, len(pos)))
        sn = random.sample(neg, min(N, len(neg)))
        fmt = bin_formatter(pred)
        mp = metric_for(sp, fmt, template)
        mn = metric_for(sn, fmt, template)
        labels = [1] * len(mp) + [0] * len(mn)
        auc = roc_auc_score(labels, mp + mn)
        pos_says_true = float(np.mean([v > 0 for v in mp]))
        neg_says_false = float(np.mean([v < 0 for v in mn]))
        print(f"{name:16} metric(True-False): pos_mean={np.mean(mp):+.2f} neg_mean={np.mean(mn):+.2f} "
              f"| sep_AUC={auc:.3f} | P(pos->True)={pos_says_true:.2f} P(neg->False)={neg_says_false:.2f}",
              flush=True)
    print("TASK_VALIDITY_DONE", flush=True)


if __name__ == "__main__":
    main()
