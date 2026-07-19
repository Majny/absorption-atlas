#!/usr/bin/env python3
"""
Probe-DIRECTION absorption — resolves the task-validity confound.

The behavioral metric (IG on True_logit-False_logit) requires the model to PERFORM the ICL task;
Gemma-2-2b barely does the binary True/False tasks (P(pos->True)~0.16), so low binary absorption
could be "weak causal target" rather than "concept not absorbed". This metric sidesteps that: it
decomposes the residual's projection onto the concept PROBE direction into SAE-latent contributions
  contribution_i = sae_act_i * (W_dec[i] . probe_dir_unit)
and asks, on candidate tokens where the main latent is SILENT, whether a single non-main latent
DOMINATES that projection (absorption) or it is spread across many (no absorption). This is purely
representational — independent of whether the model answers the task — and is applied by the SAME
code to spelling and structural properties, so the comparison is apples-to-apples.

Reads the already-computed candidate tokens + main-feature ids from each property's absorption
parquet; one forward per candidate (no IG/backward). Reports the conditioned rate (of main-silent
candidates, fraction with single-dominant probe-direction absorption) at several dominance ratios R.
"""
import argparse
import json
import os
from pathlib import Path

if "HF_TOKEN" not in os.environ:
    for cand in (Path(__file__).parent / ".env", Path.cwd() / ".env"):
        if cand.exists():
            for line in cand.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip()
            break

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})

from sae_spelling.experiments.common import DEFAULT_DEVICE, load_gemma2_model, load_gemmascope_sae  # noqa: E402
from sae_spelling.feature_absorption_calculator import EPS  # noqa: E402
from sae_spelling.prompting import (  # noqa: E402
    VERBOSE_FIRST_LETTER_TEMPLATE,
    VERBOSE_FIRST_LETTER_TOKEN_POS,
    create_icl_prompt,
    first_letter_formatter,
    last_letter_formatter,
)
from sae_spelling.vocab import LETTERS, get_alpha_tokens  # noqa: E402
from run_binary import PROPERTIES as BINPROPS, bin_formatter, word_token_pos  # noqa: E402

RESULTS = Path(__file__).parent / "results"
# letter properties: (formatter, template, word_pos, absorption-parquet)
LETTER_CFG = {
    "first_letter": (first_letter_formatter(), VERBOSE_FIRST_LETTER_TEMPLATE,
                     VERBOSE_FIRST_LETTER_TOKEN_POS, RESULTS / "gate" / "absorption_layer12_16k_l082.parquet"),
    "last_letter": (last_letter_formatter(), "{word} has the last letter:", -6,
                    RESULTS / "last_letter" / "absorption_layer12_16k_l082.parquet"),
}


def build_candidates(prop):
    """Return list of (token, probe_dir_index_or_None, main_feat_ids) + formatter/template/pos + probe."""
    if prop in LETTER_CFG:
        formatter, template, pos, pq = LETTER_CFG[prop]
        df = pd.read_parquet(pq)
        probe = torch.load(RESULTS / ("gate" if prop == "first_letter" else "last_letter")
                           / "probes" / "layer_12" / "probe.pth").cpu()
        cands = [(r["token"], LETTERS.index(r["letter"]), list(r["split_feats"])) for _, r in df.iterrows()]
        return formatter, template, pos, probe, cands
    # binary
    cfg = BINPROPS[prop]
    formatter = bin_formatter(cfg["predicate"])
    template = cfg["template"]
    df = pd.read_parquet(RESULTS / prop / "absorption_detail_l082.parquet")
    summ = json.loads((RESULTS / prop / "summary_l082.json").read_text())
    probe = torch.load(RESULTS / prop / "probes" / "layer_12" / "binary_probe.pth").cpu()
    mf = summ["main_feats"]
    cands = [(r["token"], 0, mf) for _, r in df.iterrows()]
    return formatter, template, None, probe, cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--property", required=True)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--width", type=int, default=16000)
    ap.add_argument("--l0", type=int, default=82)
    args = ap.parse_args()

    model = load_gemma2_model()
    hook = f"blocks.{args.layer}.hook_resid_post"
    sae = load_gemmascope_sae(args.layer, width=args.width, l0=args.l0)
    Wdec = sae.W_dec.detach().float().cpu()  # (d_sae, d_model), unit-norm rows (folded)

    formatter, template, pos, probe, cands = build_candidates(args.property)
    if pos is None:
        pos = word_token_pos(model, template)
    vocab = get_alpha_tokens(model.tokenizer)
    print(f"property={args.property}; candidates={len(cands)}; word_pos={pos}", flush=True)

    # precompute per-probe-row the latent contribution weights W_dec . probe_dir_unit
    probe_w = probe.weights.detach().float().cpu()  # (n_probes, d_model)
    probe_units = probe_w / probe_w.norm(dim=-1, keepdim=True)
    contrib_w = Wdec @ probe_units.T  # (d_sae, n_probes)

    Rs = [1.5, 2.0, 3.0]
    n_main_silent = 0
    n_abs = {R: 0 for R in Rs}
    # group candidates into batches sharing the same probe row + same prompt length
    with torch.inference_mode():
        for i in range(0, len(cands), 24):
            batch = cands[i:i + 24]
            prompts = [create_icl_prompt(tok, examples=vocab, base_template=template,
                                         answer_formatter=formatter, max_icl_examples=10).base
                       for tok, _, _ in batch]
            toks = model.to_tokens(prompts)
            resid = model.run_with_cache(toks, names_filter=[hook])[1][hook][:, pos, :]  # (B,d_model)
            acts = sae.encode(resid.to(sae.device)).float().cpu()  # (B,d_sae)
            for b, (tok, pidx, mf) in enumerate(batch):
                a = acts[b]
                if not bool((a[mf] < EPS).all()):
                    continue  # main not silent -> not in conditioned pool
                n_main_silent += 1
                contrib = a * contrib_w[:, pidx]  # (d_sae,) contribution to probe projection
                contrib[mf] = -float("inf")        # exclude main latents from the "absorber"
                top2 = torch.topk(contrib, 2).values
                top, second = float(top2[0]), float(top2[1])
                for R in Rs:
                    if top > 0 and top >= R * max(second, 0.0):
                        n_abs[R] += 1

    rates = {R: (n_abs[R] / n_main_silent if n_main_silent else float("nan")) for R in Rs}
    out = {"property": args.property, "n_candidates": len(cands), "n_main_silent": n_main_silent,
           "probedir_conditioned_rate": rates, "n_absorbed": n_abs}
    print(f"[probedir] {args.property}: main_silent={n_main_silent}; conditioned rate "
          + ", ".join(f"R{R}={rates[R]:.4f}" for R in Rs), flush=True)
    (RESULTS / args.property / f"probedir_l0{args.l0}.json").write_text(json.dumps(out, indent=2))
    print("PROBEDIR_DONE", flush=True)


if __name__ == "__main__":
    main()
