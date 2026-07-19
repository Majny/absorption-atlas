#!/usr/bin/env python3
"""
Solidify the mechanism: get a SMOOTH task-accuracy span at a FIXED prompt structure (k=10) by
corrupting a deterministic fraction f of the ICL example answers (wrong first letter). As f rises the
model's in-context first-letter accuracy falls smoothly; measure behavioral vs representational
absorption at each f. Clean prediction: behavioral tracks accuracy, representational stays flat.

Reuses the cached first-letter probe + k-sparse results (results/gate/...).
"""
import argparse
import hashlib
import json
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
import pandas as pd  # noqa: E402
import torch  # noqa: E402

_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})

from sae_spelling.experiments.common import (  # noqa: E402
    get_gemmascope_saes_info, get_or_make_dir, load_gemma2_model, load_gemmascope_sae, load_probe,
)
from sae_spelling.experiments.feature_absorption import (  # noqa: E402
    ABSORPTION_FEATURE_DELTA_THRESHOLD, ABSORPTION_PROBE_COS_THRESHOLD,
    _aggregate_results_df, calculate_ig_ablation_and_cos_sims, get_stats_and_likely_false_negative_tokens,
)
from sae_spelling.experiments.k_sparse_probing import (  # noqa: E402
    add_feature_splits_to_auroc_f1_df, build_f1_and_auroc_df,
    get_sparse_probing_metadata_filename, get_sparse_probing_raw_results_filename,
)
from sae_spelling.feature_absorption_calculator import EPS, FeatureAbsorptionCalculator  # noqa: E402
from sae_spelling.prompting import (  # noqa: E402
    VERBOSE_FIRST_LETTER_TEMPLATE, VERBOSE_FIRST_LETTER_TOKEN_POS,
    create_icl_prompt, first_letter_formatter,
)
from sae_spelling.vocab import LETTERS, LETTERS_UPPER, get_alpha_tokens  # noqa: E402

RESULTS = Path(__file__).parent / "results"
POS = VERBOSE_FIRST_LETTER_TOKEN_POS
TMPL = VERBOSE_FIRST_LETTER_TEMPLATE
K = 10


def corrupting_formatter(f):
    base = first_letter_formatter()

    def fmt(word):
        ans = base(word)                       # " C"
        correct = ans.strip().upper()
        h = int(hashlib.md5(word.strip().encode()).hexdigest()[:8], 16)
        if (h % 1000) < int(f * 1000):
            ci = LETTERS_UPPER.index(correct)
            return " " + LETTERS_UPPER[(ci + 1 + (h % 25)) % 26]   # deterministic wrong letter
        return ans
    return fmt


def accuracy(model, vocab, fmt, n=260):
    tok = model.tokenizer
    letter_toks = [tok.encode(f" {L}")[-1] for L in LETTERS_UPPER]
    words = random.Random(0).sample(vocab, n)
    correct = 0
    for i in range(0, len(words), 16):
        chunk = words[i:i + 16]
        prompts = [create_icl_prompt(w, examples=vocab, base_template=TMPL, answer_formatter=fmt,
                                     max_icl_examples=K).base for w in chunk]
        by = {}
        for w, p in zip(chunk, prompts):
            by.setdefault(model.to_tokens(p).shape[1], []).append((w, p))
        for _L, items in by.items():
            with torch.inference_mode():
                lg = model(model.to_tokens([p for _, p in items]))[:, -1, :]
            pred = lg[:, letter_toks].argmax(-1)
            for j, (w, _) in enumerate(items):
                correct += int(pred[j].item() == LETTERS.index(w.strip()[0].lower()))
    return correct / len(words)


def behavioral(model, sae, probe, likely_negs, sae_info, vocab, fmt, max_prompts):
    calc = FeatureAbsorptionCalculator(
        model=model, icl_word_list=vocab, max_icl_examples=K, base_template=TMPL,
        answer_formatter=fmt, word_token_pos=POS,
        probe_cos_sim_threshold=ABSORPTION_PROBE_COS_THRESHOLD,
        ablation_delta_threshold=ABSORPTION_FEATURE_DELTA_THRESHOLD,
        ig_batch_size=6, ig_interpolation_steps=6, filter_prompts_batch_size=40)
    df = calculate_ig_ablation_and_cos_sims(calc, sae, probe, likely_negs, max_prompts_per_letter=max_prompts)
    return float(_aggregate_results_df({sae_info.layer: [(df, sae_info)]})["absorption_rate"].mean())


def representational(model, sae, probe, likely_negs, vocab, fmt, contrib_w, max_prompts, R=2.0):
    hook = sae.cfg.hook_name
    n_ms, n_abs = 0, 0
    for letter, stats in likely_negs.items():
        pidx = LETTERS.index(letter)
        toks = stats.potential_false_negatives[:max_prompts]
        for i in range(0, len(toks), 24):
            chunk = toks[i:i + 24]
            prompts = [create_icl_prompt(w, examples=vocab, base_template=TMPL, answer_formatter=fmt,
                                         max_icl_examples=K).base for w in chunk]
            with torch.inference_mode():
                resid = model.run_with_cache(model.to_tokens(prompts), names_filter=[hook])[1][hook][:, POS, :]
                acts = sae.encode(resid.to(sae.device)).float().cpu()
            mf = list(stats.split_feats)
            for b in range(len(chunk)):
                a = acts[b]
                if not bool((a[mf] < EPS).all()):
                    continue
                n_ms += 1
                contrib = a * contrib_w[:, pidx]
                contrib[mf] = -float("inf")
                t2 = torch.topk(contrib, 2).values
                if float(t2[0]) > 0 and float(t2[0]) >= R * max(float(t2[1]), 0.0):
                    n_abs += 1
    return (n_abs / n_ms) if n_ms else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fracs", type=str, default="0.0,0.25,0.5,0.75,0.9")
    ap.add_argument("--max-prompts", type=int, default=20)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--width", type=int, default=16000)
    ap.add_argument("--l0", type=int, default=82)
    args = ap.parse_args()
    fracs = [float(x) for x in args.fracs.split(",")]

    model = load_gemma2_model()
    vocab = get_alpha_tokens(model.tokenizer)
    sae_info = min([s for s in get_gemmascope_saes_info(args.layer) if s.width == args.width],
                   key=lambda s: abs(s.l0 - args.l0))
    sae = load_gemmascope_sae(sae_info.layer, width=sae_info.width, l0=sae_info.l0)
    probe = load_probe(layer=args.layer, probes_dir=RESULTS / "gate" / "probes")
    pw = probe.weights.detach().float().cpu()
    contrib_w = sae.W_dec.detach().float().cpu() @ (pw / pw.norm(dim=-1, keepdim=True)).T

    sparse = RESULTS / "gate" / "k_sparse_probing"
    raw_df = pd.read_parquet(sparse / get_sparse_probing_raw_results_filename(sae_info))
    meta_df = pd.read_parquet(sparse / get_sparse_probing_metadata_filename(sae_info))
    auroc = build_f1_and_auroc_df(raw_df, meta_df)
    add_feature_splits_to_auroc_f1_df(auroc)
    likely_negs = get_stats_and_likely_false_negative_tokens(auroc, sae_info, sparse)

    rows = []
    for f in fracs:
        fmt = corrupting_formatter(f)
        acc = accuracy(model, vocab, fmt)
        beh = behavioral(model, sae, probe, likely_negs, sae_info, vocab, fmt, args.max_prompts)
        rep = representational(model, sae, probe, likely_negs, vocab, fmt, contrib_w, args.max_prompts)
        print(f"[f={f}] accuracy={acc:.3f}  behavioral={beh:.4f}  representational={rep:.3f}", flush=True)
        rows.append({"corrupt_frac": f, "accuracy": acc, "behavioral": beh, "representational": rep})

    out = get_or_make_dir(RESULTS / "icl_sweep")
    (out / "corrupt_sweep.json").write_text(json.dumps(rows, indent=2))
    print(f"saved -> {out}/corrupt_sweep.json\nCORRUPT_SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
