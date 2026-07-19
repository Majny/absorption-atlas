#!/usr/bin/env python3
"""
MECHANISM test: is behavioral (ablation) feature-absorption detection GATED by task accuracy, while
representational (projection) absorption is not? Hold model + property (first-letter, which Gemma-2-2b
CAN do and where absorption is real) FIXED; vary only the number of ICL examples k -> vary the model's
task accuracy -> measure (a) first-letter accuracy, (b) behavioral absorption rate, (c) representational
(probe-direction) absorption rate. If (b) tracks (a) while (c) is ~flat, behavioral absorption is a
task-performance artifact, not a representational fact -> a falsifiable calibration claim.

Reuses the cached first-letter probe + k-sparse results (results/gate/...).
"""
import argparse
import json
import os
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
    DEFAULT_DEVICE, get_gemmascope_saes_info, get_or_make_dir,
    load_gemma2_model, load_gemmascope_sae, load_probe,
)
from sae_spelling.experiments.feature_absorption import (  # noqa: E402
    ABSORPTION_FEATURE_DELTA_THRESHOLD, ABSORPTION_PROBE_COS_THRESHOLD,
    _aggregate_results_df, calculate_ig_ablation_and_cos_sims,
    get_stats_and_likely_false_negative_tokens, letter_delta_metric,
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
FMT = first_letter_formatter()


def first_letter_accuracy(model, vocab, k, n=260):
    """fraction of sampled words whose correct capital-letter token is argmax among the 26 letters."""
    tok = model.tokenizer
    letter_toks = [tok.encode(f" {L}")[-1] for L in LETTERS_UPPER]
    import random
    words = random.Random(0).sample(vocab, n)
    correct = 0
    for i in range(0, len(words), 16):
        chunk = words[i:i + 16]
        prompts = [create_icl_prompt(w, examples=vocab, base_template=TMPL, answer_formatter=FMT,
                                     max_icl_examples=k).base for w in chunk]
        by = {}
        for w, p in zip(chunk, prompts):
            by.setdefault(model.to_tokens(p).shape[1], []).append((w, p))
        for _L, items in by.items():
            with torch.inference_mode():
                lg = model(model.to_tokens([p for _, p in items]))[:, -1, :]
            sub = lg[:, letter_toks]  # (B,26)
            pred = sub.argmax(-1)
            for j, (w, _) in enumerate(items):
                gold = LETTERS.index(w.strip()[0].lower())
                correct += int(pred[j].item() == gold)
    return correct / len(words)


def behavioral_rate(model, sae, probe, likely_negs, sae_info, vocab, k, max_prompts):
    calc = FeatureAbsorptionCalculator(
        model=model, icl_word_list=vocab, max_icl_examples=k if k > 0 else None,
        base_template=TMPL, answer_formatter=FMT, word_token_pos=POS,
        probe_cos_sim_threshold=ABSORPTION_PROBE_COS_THRESHOLD,
        ablation_delta_threshold=ABSORPTION_FEATURE_DELTA_THRESHOLD,
        ig_batch_size=6, ig_interpolation_steps=6, filter_prompts_batch_size=40)
    # k=0 -> max_icl_examples None means "all"; force true 0-shot by passing 0 via a tiny shim:
    if k == 0:
        calc.max_icl_examples = 0
    df = calculate_ig_ablation_and_cos_sims(calc, sae, probe, likely_negs,
                                            max_prompts_per_letter=max_prompts)
    agg = _aggregate_results_df({sae_info.layer: [(df, sae_info)]})
    return float(agg["absorption_rate"].mean())


def representational_rate(model, sae, probe, likely_negs, vocab, k, Wdec, contrib_w, max_prompts, R=2.0):
    hook = f"blocks.{sae.cfg.hook_name.split('.')[1]}.hook_resid_post"
    n_ms, n_abs = 0, 0
    for letter, stats in likely_negs.items():
        pidx = LETTERS.index(letter)
        toks = stats.potential_false_negatives[:max_prompts]
        for i in range(0, len(toks), 24):
            chunk = toks[i:i + 24]
            prompts = [create_icl_prompt(w, examples=vocab, base_template=TMPL, answer_formatter=FMT,
                                         max_icl_examples=k if k > 0 else 0).base for w in chunk]
            with torch.inference_mode():
                resid = model.run_with_cache(model.to_tokens(prompts),
                                             names_filter=[hook])[1][hook][:, POS, :]
                acts = sae.encode(resid.to(sae.device)).float().cpu()
            for b in range(len(chunk)):
                a = acts[b]
                mf = list(stats.split_feats)
                if not bool((a[mf] < EPS).all()):
                    continue
                n_ms += 1
                contrib = a * contrib_w[:, pidx]
                contrib[mf] = -float("inf")
                t2 = torch.topk(contrib, 2).values
                top, sec = float(t2[0]), float(t2[1])
                if top > 0 and top >= R * max(sec, 0.0):
                    n_abs += 1
    return (n_abs / n_ms) if n_ms else float("nan"), n_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=str, default="0,1,3,10")
    ap.add_argument("--max-prompts", type=int, default=20)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--width", type=int, default=16000)
    ap.add_argument("--l0", type=int, default=82)
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]

    model = load_gemma2_model()
    vocab = get_alpha_tokens(model.tokenizer)
    sae_info = min([s for s in get_gemmascope_saes_info(args.layer) if s.width == args.width],
                   key=lambda s: abs(s.l0 - args.l0))
    sae = load_gemmascope_sae(sae_info.layer, width=sae_info.width, l0=sae_info.l0)
    probe = load_probe(layer=args.layer, probes_dir=RESULTS / "gate" / "probes")
    Wdec = sae.W_dec.detach().float().cpu()
    pw = probe.weights.detach().float().cpu()
    contrib_w = Wdec @ (pw / pw.norm(dim=-1, keepdim=True)).T

    sparse = RESULTS / "gate" / "k_sparse_probing"
    raw_df = pd.read_parquet(sparse / get_sparse_probing_raw_results_filename(sae_info))
    meta_df = pd.read_parquet(sparse / get_sparse_probing_metadata_filename(sae_info))
    auroc = build_f1_and_auroc_df(raw_df, meta_df)
    add_feature_splits_to_auroc_f1_df(auroc)
    likely_negs = get_stats_and_likely_false_negative_tokens(auroc, sae_info, sparse)

    rows = []
    for k in ks:
        acc = first_letter_accuracy(model, vocab, k)
        beh = behavioral_rate(model, sae, probe, likely_negs, sae_info, vocab, k, args.max_prompts)
        rep, nms = representational_rate(model, sae, probe, likely_negs, vocab, k, Wdec, contrib_w,
                                         args.max_prompts)
        print(f"[k={k}] first_letter_acc={acc:.3f}  behavioral_absorption={beh:.4f}  "
              f"representational={rep:.3f} (n_main_silent={nms})", flush=True)
        rows.append({"k": k, "accuracy": acc, "behavioral": beh, "representational": rep, "n_ms": nms})

    out = get_or_make_dir(RESULTS / "icl_sweep")
    (out / "sweep.json").write_text(json.dumps(rows, indent=2))
    print(f"saved -> {out}/sweep.json\nICL_SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
