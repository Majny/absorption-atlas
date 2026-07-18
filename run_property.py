#!/usr/bin/env python3
"""
Multi-property absorption driver — does feature absorption appear for token properties OTHER than
"first letter"? Generalizes run_gate.py over a registry of properties.

For 26-class LETTER properties (first_letter, last_letter, nth-letter), the entire sae-spelling
pipeline is reusable unchanged: the probe is 26-class (LETTERS), the k-sparse machinery and the
IG metric (`letter_delta_metric`, target-letter logit vs the other 25) are letter-generic. Only
three things vary per property: the answer FORMATTER, the prompt TEMPLATE, and the word-token
POSITION in that template. Each property gets its own probe dir so probes don't collide.

(Binary properties — is_capitalized / is_numeric / has_suffix — need a separate single-class
orchestration and are NOT handled here yet; see PLAN.md.)

Usage:
  python run_property.py --property last_letter --layer 12 --width 16000 --target-l0 82 --max-prompts 200
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

import pandas as pd  # noqa: E402
import torch  # noqa: E402

# torch 2.6+ flips torch.load weights_only default to True, breaking the LinearProbe checkpoint.
_orig_torch_load = torch.load
def _torch_load_compat(*a, **k):  # noqa: E306
    k.setdefault("weights_only", False)
    return _orig_torch_load(*a, **k)
torch.load = _torch_load_compat

from sae_spelling.experiments.common import (  # noqa: E402
    DEFAULT_DEVICE,
    SaeInfo,
    create_and_train_probe,
    get_gemmascope_saes_info,
    get_or_make_dir,
    load_gemma2_model,
    load_gemmascope_sae,
    load_probe,
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
    first_letter_formatter,
    last_letter_formatter,
)
from sae_spelling.vocab import get_alpha_tokens  # noqa: E402

# Each LETTER property: (answer formatter, prompt template, word-token position in that template).
# The word position is verified by tokenizing the template (see verify_positions()).
PROPERTIES = {
    "first_letter": (first_letter_formatter(), "{word} has the first letter:", -6),
    "last_letter": (last_letter_formatter(), "{word} has the last letter:", -6),
}


def pick_sae_info(layer: int, width: int, target_l0: int) -> SaeInfo:
    infos = [s for s in get_gemmascope_saes_info(layer) if s.width == width]
    if not infos:
        raise SystemExit(f"No SAE for layer={layer} width={width}")
    print(f"  available L0s @ layer{layer}/{width//1000}k: {sorted(s.l0 for s in infos)}")
    return min(infos, key=lambda s: abs(s.l0 - target_l0))


def verify_position(model, base_template: str, expected_pos: int):
    """Fail loudly if the word token isn't where the property config claims."""
    toks = model.to_str_tokens(base_template.format(word="cat"))
    # the word 'cat' should be a single token sitting at expected_pos
    got = toks[expected_pos].strip()
    print(f"  template {base_template!r}: token[{expected_pos}]={got!r} (tokens={len(toks)})")
    if got.lower() not in ("cat",):
        raise SystemExit(
            f"word-position check FAILED: token[{expected_pos}]={got!r}, expected 'cat'. "
            f"Fix the position in PROPERTIES for this template.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--property", required=True, choices=list(PROPERTIES))
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--width", type=int, default=16000)
    ap.add_argument("--target-l0", type=int, default=82)
    ap.add_argument("--max-prompts", type=int, default=200)
    ap.add_argument("--results-dir", type=str,
                    default=str(Path(__file__).parent / "results"))
    args = ap.parse_args()
    formatter, base_template, token_pos = PROPERTIES[args.property]

    print(f"device = {DEFAULT_DEVICE}; property = {args.property}", flush=True)
    prop_dir = get_or_make_dir(Path(args.results_dir) / args.property)
    probes_dir = get_or_make_dir(prop_dir / "probes")
    sparse_dir = get_or_make_dir(prop_dir / "k_sparse_probing")

    model = load_gemma2_model()
    print(f"model loaded (n_layers={model.cfg.n_layers})", flush=True)
    verify_position(model, base_template, token_pos)

    sae_info = pick_sae_info(args.layer, args.width, args.target_l0)
    print(f"  -> using {sae_info}", flush=True)

    # 1) property-specific 26-class probe (cached)
    probe_path = Path(probes_dir) / f"layer_{args.layer}" / "probe.pth"
    if not probe_path.exists():
        print(f"training {args.property} probe...", flush=True)
        create_and_train_probe(
            model=model,
            formatter=formatter,
            hook_point=f"blocks.{args.layer}.hook_resid_post",
            probes_dir=probes_dir,
            vocab=get_alpha_tokens(model.tokenizer),
            batch_size=64, num_epochs=50, lr=1e-2,
            device=torch.device(DEFAULT_DEVICE),
            base_template=base_template,
            pos_idx=token_pos,
        )
    else:
        print("probe cached, skipping", flush=True)

    # 2) k-sparse probing (cached raw parquet where absorption expects it)
    raw_path = Path(sparse_dir) / get_sparse_probing_raw_results_filename(sae_info)
    meta_path = Path(sparse_dir) / get_sparse_probing_metadata_filename(sae_info)
    if not raw_path.exists():
        print("running k-sparse probing...", flush=True)
        raw_df, meta_df = load_and_run_eval_probe_and_sae_k_sparse_raw_scores(
            sae_info, model, probes_dir)
        raw_df.to_parquet(raw_path, index=False)
        meta_df.to_parquet(meta_path, index=False)
    else:
        raw_df = pd.read_parquet(raw_path)
        meta_df = pd.read_parquet(meta_path)
        print("k-sparse cached, skipping", flush=True)
    auroc_f1_df = build_f1_and_auroc_df(raw_df, meta_df)
    add_feature_splits_to_auroc_f1_df(auroc_f1_df)

    # 3) absorption via IG ablation (letter_delta_metric is letter-generic -> reused as-is)
    calculator = FeatureAbsorptionCalculator(
        model=model,
        icl_word_list=get_alpha_tokens(model.tokenizer),
        max_icl_examples=10,
        base_template=base_template,
        answer_formatter=formatter,
        word_token_pos=token_pos,
        probe_cos_sim_threshold=ABSORPTION_PROBE_COS_THRESHOLD,
        ablation_delta_threshold=ABSORPTION_FEATURE_DELTA_THRESHOLD,
        ig_batch_size=6, ig_interpolation_steps=6, filter_prompts_batch_size=40,
    )
    probe = load_probe(layer=args.layer, probes_dir=probes_dir)
    sae = load_gemmascope_sae(sae_info.layer, width=sae_info.width, l0=sae_info.l0)
    likely_negs = get_stats_and_likely_false_negative_tokens(auroc_f1_df, sae_info, Path(sparse_dir))
    print(f"running IG-ablation absorption (max {args.max_prompts}/letter)...", flush=True)
    abs_df = calculate_ig_ablation_and_cos_sims(
        calculator, sae, probe, likely_negs, max_prompts_per_letter=args.max_prompts)
    abs_df.to_parquet(
        prop_dir / f"absorption_layer{args.layer}_{args.width//1000}k_l0{sae_info.l0}.parquet",
        index=False)

    # 4) aggregate
    agg = _aggregate_results_df({args.layer: [(abs_df, sae_info)]})
    mean_rate = float(agg["absorption_rate"].mean())
    per_letter = agg[["letter", "absorption_rate", "num_absorption",
                      "num_probe_true_positives"]].sort_values("absorption_rate", ascending=False)
    print(f"\n===== {args.property.upper()} ABSORPTION =====", flush=True)
    print(f"SAE: layer {sae_info.layer}, {args.width//1000}k, L0 {sae_info.l0}", flush=True)
    print(f"MEAN absorption rate: {mean_rate:.4f}", flush=True)
    print(per_letter.to_string(index=False), flush=True)

    summary = {
        "property": args.property, "layer": sae_info.layer, "width": sae_info.width,
        "l0": sae_info.l0, "max_prompts_per_letter": args.max_prompts,
        "mean_absorption_rate": mean_rate,
        "per_letter": per_letter.to_dict(orient="records"),
        "device": DEFAULT_DEVICE,
    }
    out = prop_dir / f"summary_l0{sae_info.l0}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved -> {out}\nPROPERTY_DONE", flush=True)


if __name__ == "__main__":
    main()
