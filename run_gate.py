#!/usr/bin/env python3
"""
Week-1 GATE: reproduce first-letter feature absorption on ONE current Gemma Scope SAE.

Pipeline (single SAE, not the full sweep):
  1. train + cache the layer LR probe (ground-truth concept direction)
  2. k-sparse probing -> find the "main"/split SAE latents, save raw parquet
  3. IG-ablation absorption calculator on candidate false-negatives
  4. aggregate -> per-letter absorption_rate + mean

Portable: device is left to the sae-spelling library defaults (cpu on Mac, cuda on a
GPU box), so model + SAE always live on the same device. Everything is cached, so reruns
are cheap and a killed run resumes.

Usage:
  python run_gate.py --layer 12 --width 16000 --target-l0 68 --max-prompts 200
"""
import argparse
import json
import os
import time
from pathlib import Path

# --- HF token: prefer env (cluster), else the gitignored local .env ---
if "HF_TOKEN" not in os.environ:
    for cand in (Path(__file__).parent / ".env", Path.cwd() / ".env"):
        if cand.exists():
            for line in cand.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip()
            break

import pandas as pd  # noqa: E402
import torch  # noqa: E402

# sae-spelling was written for torch <2.6 (weights_only default False). torch 2.6+ flips the
# default to True, which refuses to unpickle the custom LinearProbe checkpoint. Our probe files
# are locally produced + trusted, so restore the old behavior for all torch.load calls.
_orig_torch_load = torch.load
def _torch_load_compat(*a, **k):  # noqa: E306
    k.setdefault("weights_only", False)
    return _orig_torch_load(*a, **k)
torch.load = _torch_load_compat

from sae_spelling.experiments.common import (  # noqa: E402
    DEFAULT_DEVICE,
    SaeInfo,
    get_gemmascope_saes_info,
    get_or_make_dir,
    load_gemma2_model,
    load_gemmascope_sae,
    load_probe,
    train_and_save_probes,
)
from sae_spelling.experiments.feature_absorption import (  # noqa: E402
    ABSORPTION_FEATURE_DELTA_THRESHOLD,
    ABSORPTION_PROBE_COS_THRESHOLD,
    _aggregate_results_df,
    calculate_ig_ablation_and_cos_sims,
    get_stats_and_likely_false_negative_tokens,
)
from sae_spelling.experiments.k_sparse_probing import (  # noqa: E402
    add_feature_splits_to_auroc_f1_df,
    build_f1_and_auroc_df,
    get_sparse_probing_metadata_filename,
    get_sparse_probing_raw_results_filename,
    load_and_run_eval_probe_and_sae_k_sparse_raw_scores,
)
from sae_spelling.feature_absorption_calculator import FeatureAbsorptionCalculator  # noqa: E402
from sae_spelling.prompting import (  # noqa: E402
    VERBOSE_FIRST_LETTER_TEMPLATE,
    VERBOSE_FIRST_LETTER_TOKEN_POS,
    first_letter_formatter,
)
from sae_spelling.vocab import get_alpha_tokens  # noqa: E402


def pick_sae_info(layer: int, width: int, target_l0: int) -> SaeInfo:
    infos = [s for s in get_gemmascope_saes_info(layer) if s.width == width]
    if not infos:
        raise SystemExit(f"No SAE for layer={layer} width={width}")
    best = min(infos, key=lambda s: abs(s.l0 - target_l0))
    print(f"  available L0s @ layer{layer}/{width//1000}k: {sorted(s.l0 for s in infos)}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--width", type=int, default=16000)
    ap.add_argument("--target-l0", type=int, default=68)
    ap.add_argument("--max-prompts", type=int, default=200, help="IG samples per letter cap")
    ap.add_argument("--results-dir", type=str,
                    default=str(Path(__file__).parent / "results" / "gate"))
    args = ap.parse_args()

    print(f"device (library default) = {DEFAULT_DEVICE}", flush=True)
    results_dir = get_or_make_dir(args.results_dir)
    probes_dir = get_or_make_dir(Path(args.results_dir) / "probes")
    sparse_dir = get_or_make_dir(Path(args.results_dir) / "k_sparse_probing")

    t0 = time.time()
    model = load_gemma2_model()
    print(f"[{time.time()-t0:.0f}s] model loaded (n_layers={model.cfg.n_layers})", flush=True)

    sae_info = pick_sae_info(args.layer, args.width, args.target_l0)
    print(f"  -> using {sae_info}", flush=True)

    # 1) probe (cached). Pre-train explicitly so the cuda-hardcoded auto-train path never fires.
    probe_path = Path(probes_dir) / f"layer_{args.layer}" / "probe.pth"
    if not probe_path.exists():
        t = time.time()
        print("training LR probe over alpha-token vocab...", flush=True)
        train_and_save_probes(model, [args.layer], probes_dir,
                              device=torch.device(DEFAULT_DEVICE))
        print(f"[{time.time()-t:.0f}s] probe trained", flush=True)
    else:
        print("probe cached, skipping", flush=True)

    # 2) k-sparse probing for this ONE SAE; save raw parquet where absorption expects it
    raw_path = Path(sparse_dir) / get_sparse_probing_raw_results_filename(sae_info)
    meta_path = Path(sparse_dir) / get_sparse_probing_metadata_filename(sae_info)
    if not raw_path.exists():
        t = time.time()
        print("running k-sparse probing...", flush=True)
        raw_df, meta_df = load_and_run_eval_probe_and_sae_k_sparse_raw_scores(
            sae_info, model, probes_dir)
        raw_df.to_parquet(raw_path, index=False)
        meta_df.to_parquet(meta_path, index=False)
        print(f"[{time.time()-t:.0f}s] k-sparse done ({len(raw_df)} eval tokens)", flush=True)
    else:
        raw_df = pd.read_parquet(raw_path)
        meta_df = pd.read_parquet(meta_path)
        print("k-sparse cached, skipping", flush=True)
    auroc_f1_df = build_f1_and_auroc_df(raw_df, meta_df)
    add_feature_splits_to_auroc_f1_df(auroc_f1_df)

    # 3) absorption via IG ablation on candidate false-negatives (max-prompts controllable)
    vocab = get_alpha_tokens(model.tokenizer)
    calculator = FeatureAbsorptionCalculator(
        model=model,
        icl_word_list=vocab,
        max_icl_examples=10,
        base_template=VERBOSE_FIRST_LETTER_TEMPLATE,
        answer_formatter=first_letter_formatter(),
        word_token_pos=VERBOSE_FIRST_LETTER_TOKEN_POS,
        probe_cos_sim_threshold=ABSORPTION_PROBE_COS_THRESHOLD,
        ablation_delta_threshold=ABSORPTION_FEATURE_DELTA_THRESHOLD,
        ig_batch_size=6,
        ig_interpolation_steps=6,
        filter_prompts_batch_size=40,
    )
    probe = load_probe(layer=args.layer, probes_dir=probes_dir)
    sae = load_gemmascope_sae(sae_info.layer, width=sae_info.width, l0=sae_info.l0)
    likely_negs = get_stats_and_likely_false_negative_tokens(auroc_f1_df, sae_info, Path(sparse_dir))
    t = time.time()
    print(f"running IG-ablation absorption (max {args.max_prompts}/letter)...", flush=True)
    abs_df = calculate_ig_ablation_and_cos_sims(
        calculator, sae, probe, likely_negs, max_prompts_per_letter=args.max_prompts)
    print(f"[{time.time()-t:.0f}s] absorption done ({len(abs_df)} sampled tokens)", flush=True)
    abs_path = Path(results_dir) / f"absorption_layer{args.layer}_{args.width//1000}k_l0{sae_info.l0}.parquet"
    abs_df.to_parquet(abs_path, index=False)

    # 4) aggregate
    agg = _aggregate_results_df({args.layer: [(abs_df, sae_info)]})
    mean_rate = float(agg["absorption_rate"].mean())
    per_letter = agg[["letter", "absorption_rate", "num_absorption",
                      "num_probe_true_positives"]].sort_values("absorption_rate", ascending=False)
    print("\n===== ABSORPTION GATE RESULT =====", flush=True)
    print(f"SAE: layer {sae_info.layer}, width {args.width//1000}k, L0 {sae_info.l0}", flush=True)
    print(f"MEAN absorption rate (over letters): {mean_rate:.3f}", flush=True)
    print(per_letter.to_string(index=False), flush=True)

    summary = {
        "layer": sae_info.layer, "width": sae_info.width, "l0": sae_info.l0,
        "max_prompts_per_letter": args.max_prompts,
        "mean_absorption_rate": mean_rate,
        "per_letter": per_letter.to_dict(orient="records"),
        "thresholds": {"probe_cos": ABSORPTION_PROBE_COS_THRESHOLD,
                       "ablation_delta": ABSORPTION_FEATURE_DELTA_THRESHOLD},
        "device": DEFAULT_DEVICE,
    }
    summary_path = Path(results_dir) / f"gate_summary_l0{sae_info.l0}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved -> {summary_path}", flush=True)
    print("GATE_DONE", flush=True)


if __name__ == "__main__":
    main()
