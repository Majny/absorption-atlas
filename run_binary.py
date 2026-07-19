#!/usr/bin/env python3
"""
run_binary.py — feature absorption for BINARY, structurally-different token properties
(is-capitalized, is-numeric, has-suffix). This is the real test of generality the field's open
question asks for: does absorption appear for properties UNRELATED to single-letter spelling?

The character pipeline is 26-class; binary properties need a single-class orchestration:
one binary residual probe, one binary logit-diff IG metric, a 1-class k-sparse + false-negative
flow. The property-agnostic FeatureAbsorptionCalculator.calculate_absorption_sampled() is reused.

HONEST REPORTING (the whole point — a number alone is not trustworthy here):
  (a) binary residual-probe AUROC     -> does the concept even have a linear direction?
  (b) k-sparse F1-vs-k curve + best F1 -> does a CLEAN "main" SAE latent even exist? If not,
      absorption is ill-defined for this property (that is itself a finding).
  (c) absorption rate = #absorbed / #probe-true-positives
  (d) IG-gap threshold SENSITIVITY     -> the 1.0 gap is calibrated to the 26-letter metric; the
      binary metric has a different IG scale, so we report the rate across a threshold sweep
      instead of hard-copying 1.0.

Usage:
  python run_binary.py --property is_capitalized --layer 12 --width 16000 --target-l0 82 --max-prompts 200
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

import random  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

_orig_torch_load = torch.load
def _torch_load_compat(*a, **k):  # noqa: E306
    k.setdefault("weights_only", False)
    return _orig_torch_load(*a, **k)
torch.load = _torch_load_compat

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402

from sae_spelling.experiments.common import (  # noqa: E402
    DEFAULT_DEVICE,
    SaeInfo,
    get_gemmascope_saes_info,
    get_or_make_dir,
    load_gemma2_model,
    load_gemmascope_sae,
)
from sae_spelling.experiments.k_sparse_probing import KS, _get_sae_acts, train_sparse_multi_probe  # noqa: E402
from sae_spelling.feature_absorption_calculator import EPS, FeatureAbsorptionCalculator  # noqa: E402
from sae_spelling.probing import (  # noqa: E402
    create_dataset_probe_training,
    gen_and_save_df_acts_probing,
    train_binary_probe,
)
from sae_spelling.vocab import get_alpha_tokens  # noqa: E402

FEATURE_SPLIT_F1_JUMP = 0.03


def _strip(w: str) -> str:
    return w.strip()


# Each binary, NON-character property: predicate(word)->bool + prompt template. Vocab is the same
# Latin-alpha token set as the letter pipeline (get_alpha_tokens) so labels are clean and comparable;
# is-numeric was dropped because Gemma's vocab has ~0 pure-ASCII-digit single tokens (a real
# label-availability limit, itself a finding). The answer is " True"/" False".
PROPERTIES = {
    "is_capitalized": dict(
        predicate=lambda w: _strip(w)[:1].isupper(),
        template="{word} starts with a capital letter:",
    ),
    "all_caps": dict(
        predicate=lambda w: len(_strip(w)) > 1 and _strip(w).isupper(),
        template="{word} is in all capital letters:",
    ),
    "suffix_ing": dict(
        predicate=lambda w: _strip(w).lower().endswith("ing"),
        template="{word} ends with -ing:",
    ),
    "suffix_ed": dict(
        predicate=lambda w: _strip(w).lower().endswith("ed"),
        template="{word} ends with -ed:",
    ),
}


def bin_formatter(predicate, prefix=" "):
    def f(word):
        return prefix + ("True" if predicate(word) else "False")
    return f


def binary_metric_fn(tokenizer):
    # the True/False answer must each be a SINGLE token, else the logit-diff metric is meaningless
    for ans in (" True", " False"):
        if len(tokenizer.encode(ans, add_special_tokens=False)) != 1:
            raise SystemExit(f"answer {ans!r} is not a single token — binary metric invalid")
    pos_tok = tokenizer.encode(" True", add_special_tokens=False)[-1]
    neg_tok = tokenizer.encode(" False", add_special_tokens=False)[-1]

    def metric_fn(logits):
        return logits[:, -1, pos_tok] - logits[:, -1, neg_tok]
    return metric_fn


def word_token_pos(model, template: str) -> int:
    """Negative index of the (single-token) word in the base template; asserts consistency."""
    toks = model.to_str_tokens(template.format(word="cat"))
    idxs = [i for i, t in enumerate(toks) if t.strip() == "cat"]
    if len(idxs) != 1:
        raise SystemExit(f"word not a clean single token in {template!r}: {toks}")
    return idxs[0] - len(toks)  # negative index from the end


def pick_sae_info(layer, width, target_l0) -> SaeInfo:
    infos = [s for s in get_gemmascope_saes_info(layer) if s.width == width]
    if not infos:
        raise SystemExit(f"No SAE for layer={layer} width={width}")
    print(f"  L0s @ L{layer}/{width//1000}k: {sorted(s.l0 for s in infos)}")
    return min(infos, key=lambda s: abs(s.l0 - target_l0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--property", required=True, choices=list(PROPERTIES))
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--width", type=int, default=16000)
    ap.add_argument("--target-l0", type=int, default=82)
    ap.add_argument("--max-prompts", type=int, default=200)
    ap.add_argument("--results-dir", type=str, default=str(Path(__file__).parent / "results"))
    args = ap.parse_args()
    cfg = PROPERTIES[args.property]
    predicate, template = cfg["predicate"], cfg["template"]

    print(f"device={DEFAULT_DEVICE}; property={args.property}; template={template!r}", flush=True)
    prop_dir = get_or_make_dir(Path(args.results_dir) / args.property)
    probes_dir = get_or_make_dir(prop_dir / "probes")

    model = load_gemma2_model()
    tokenizer = model.tokenizer
    pos = word_token_pos(model, template)
    print(f"model loaded; word_token_pos={pos}", flush=True)

    sae_info = pick_sae_info(args.layer, args.width, args.target_l0)
    print(f"  -> {sae_info}", flush=True)
    hook = f"blocks.{args.layer}.hook_resid_post"

    # ---- 1) binary residual probe (concept direction) ----
    formatter = bin_formatter(predicate)
    vocab = get_alpha_tokens(tokenizer)  # Latin a-zA-Z single tokens, same set as the letter pipeline
    n_pos = sum(predicate(w) for w in vocab)
    print(f"vocab={len(vocab)} tokens, positives={n_pos} ({100*n_pos/len(vocab):.1f}%)", flush=True)
    if n_pos < 50:
        raise SystemExit(f"too few positive tokens ({n_pos}) for {args.property}")

    cache = Path(probes_dir) / f"layer_{args.layer}"
    d_model = model.cfg.d_model
    if (cache / "train_act_tensor.dat").exists() and (cache / "test_act_tensor.dat").exists():
        print("activations cached, loading", flush=True)
        train_df = pd.read_csv(cache / "train_df.csv", keep_default_na=False, na_values=[""])
        test_df = pd.read_csv(cache / "test_df.csv", keep_default_na=False, na_values=[""])
        train_acts = np.memmap(cache / "train_act_tensor.dat", dtype="float32", mode="r",
                               shape=(len(train_df), d_model))
        test_acts = np.memmap(cache / "test_act_tensor.dat", dtype="float32", mode="r",
                              shape=(len(test_df), d_model))
    else:
        train_ds, test_ds = create_dataset_probe_training(
            vocab=vocab, formatter=formatter, num_prompts_per_token=1, base_template=template,
            answer_class_fn=lambda ans: 1 if ans.strip() == "True" else 0)
        train_df, test_df, train_acts, test_acts = gen_and_save_df_acts_probing(
            model=model, train_dataset=train_ds, test_dataset=test_ds, path=probes_dir,
            hook_point=hook, layer=args.layer, batch_size=64, position_idx=pos)

    dev = torch.device(DEFAULT_DEVICE)
    Xtr = torch.from_numpy(np.asarray(train_acts)).float()
    Xte = torch.from_numpy(np.asarray(test_acts)).float()
    ytr = torch.tensor(train_df["answer_class"].values).long()
    yte = torch.tensor(test_df["answer_class"].values).long()
    probe_pth = cache / "binary_probe.pth"
    if probe_pth.exists():
        print("probe cached, loading", flush=True)
        probe = torch.load(probe_pth).cpu()
    else:
        probe = train_binary_probe(Xtr.to(dev), ytr.float().to(dev), num_epochs=50, device=dev).cpu()
        torch.save(probe, probe_pth)
    probe_dir = probe.weights[0].detach()  # (d_model,)

    with torch.no_grad():
        te_logits = probe(Xte).squeeze(-1)
    probe_auroc = float(roc_auc_score(yte.numpy(), te_logits.numpy()))
    print(f"[probe] binary residual-probe test AUROC = {probe_auroc:.3f}", flush=True)

    # ---- 2) 1-class k-sparse probing: does a CLEAN main latent exist? ----
    sae = load_gemmascope_sae(sae_info.layer, width=sae_info.width, l0=sae_info.l0)
    sae_tr = _get_sae_acts(sae, Xtr, sae_post_act=True)   # (Ntr, d_sae)
    sae_te = _get_sae_acts(sae, Xte, sae_post_act=True)   # (Nte, d_sae)
    l1 = train_sparse_multi_probe(
        sae_tr.to(sae.device), ytr.unsqueeze(1).to(sae.device), num_probes=1,
        l1_decay=0.01, num_epochs=50, device=sae.device).float().cpu()
    ranked = l1.weights[0].topk(max(KS)).indices  # by signed weight (predictive of positive)

    f1_by_k, feats_by_k, clf_by_k = {}, {}, {}
    ytr_np, yte_np = ytr.numpy(), yte.numpy()
    for k in KS:
        feats = ranked[:k]
        clf = LogisticRegression(max_iter=500, class_weight="balanced").fit(
            sae_tr[:, feats].float().numpy(), ytr_np)
        f1_by_k[k] = float(f1_score(yte_np, clf.predict(sae_te[:, feats].float().numpy())))
        feats_by_k[k], clf_by_k[k] = feats, clf

    # feature-split detection: keep extending k while F1 jumps >= 0.03
    best, main_k = -1.0, KS[0]
    for k in [kk for kk in KS if kk <= 15]:
        if f1_by_k[k] > best + FEATURE_SPLIT_F1_JUMP:
            best, main_k = f1_by_k[k], k
        else:
            break
    main_feats = feats_by_k[main_k]
    main_clf = clf_by_k[main_k]
    best_f1 = max(f1_by_k.values())
    clean_latent = best_f1 >= 0.5  # if even the best k-sparse probe is poor, absorption is ill-defined
    print(f"[k-sparse] best F1={best_f1:.3f} (k*={main_k}, main_feats={main_feats.tolist()}); "
          f"clean_latent={clean_latent}", flush=True)
    print(f"[k-sparse] F1-vs-k: " + ", ".join(f"{k}:{f1_by_k[k]:.2f}" for k in KS if k <= 15), flush=True)

    # ---- 3) candidate false-negatives: probe fires (+1) but main latents miss ----
    main_pred_te = main_clf.decision_function(sae_te[:, main_feats].float().numpy())
    probe_pos = te_logits.numpy() > 0
    is_true = yte_np == 1
    probe_true_positives = int((probe_pos & is_true).sum())
    cand_mask = is_true & probe_pos & (main_pred_te <= 0)
    cand_tokens = [test_df.iloc[i]["token"] for i in np.where(cand_mask)[0]]
    print(f"[candidates] probe_TP={probe_true_positives}, "
          f"false-negative candidates={len(cand_tokens)}", flush=True)

    result = {
        "property": args.property, "template": template, "layer": sae_info.layer,
        "width": sae_info.width, "l0": sae_info.l0, "word_token_pos": pos,
        "vocab_size": len(vocab), "positives": int(n_pos),
        "probe_auroc": probe_auroc, "ksparse_best_f1": best_f1, "ksparse_main_k": int(main_k),
        "ksparse_f1_by_k": {int(k): f1_by_k[k] for k in KS},
        "main_feats": main_feats.tolist(), "clean_latent": bool(clean_latent),
        "probe_true_positives": probe_true_positives, "num_candidates": len(cand_tokens),
        "device": DEFAULT_DEVICE, "max_prompts_per_property": args.max_prompts,
    }

    if not clean_latent or len(cand_tokens) == 0:
        result["absorption_rate"] = None
        result["note"] = ("no clean main latent — absorption ill-defined" if not clean_latent
                          else "no false-negative candidates")
        print(f"[absorption] SKIPPED: {result['note']}", flush=True)
    else:
        # ---- 4) IG-ablation absorption via the property-agnostic calculator ----
        calc = FeatureAbsorptionCalculator(
            model=model, icl_word_list=vocab, max_icl_examples=10, base_template=template,
            answer_formatter=formatter, word_token_pos=pos,
            probe_cos_sim_threshold=0.025, ablation_delta_threshold=0.0,  # capture all; sweep in post
            ig_batch_size=6, ig_interpolation_steps=6, filter_prompts_batch_size=40)
        sampled = calc.calculate_absorption_sampled(
            sae, words=cand_tokens, probe_dir=probe_dir, metric_fn=binary_metric_fn(tokenizer),
            main_feature_ids=main_feats.tolist(), max_ablation_samples=args.max_prompts)
        portion = sampled.sample_portion

        # IG-gap threshold sensitivity: recompute is_absorption at several gaps from stored scores
        gaps = [0.5, 1.0, 2.0, 4.0]
        rates = {}
        for g in gaps:
            n_abs = 0
            for s in sampled.sample_results:
                main_silent = all(fs.activation < EPS for fs in s.main_feature_scores)
                top = s.top_ablation_feature_scores[0]
                second = s.top_ablation_feature_scores[1]
                gap = abs(top.ablation_score) - abs(second.ablation_score)
                if (main_silent and top.ablation_score < 0 and gap >= g
                        and top.probe_cos_sim >= 0.025):
                    n_abs += 1
            rates[g] = (n_abs / portion) / probe_true_positives if probe_true_positives else 0.0
        # diagnostic funnel: WHERE do candidates drop out of the absorption criteria?
        # (distinguishes a REAL absence of absorption from a metric artifact / distributed absorption)
        srs = sampled.sample_results
        ms = [s for s in srs if all(fs.activation < EPS for fs in s.main_feature_scores)]
        tops = [s.top_ablation_feature_scores[0] for s in ms]
        seconds = [s.top_ablation_feature_scores[1] for s in ms]
        funnel = {
            "sampled": len(srs),
            "main_silent": len(ms),
            "top_ablation_negative": int(sum(t.ablation_score < 0 for t in tops)),
            "top_cos_ge_thresh": int(sum(t.probe_cos_sim >= 0.025 for t in tops)),
            "neg_and_cos_ok": int(sum(t.ablation_score < 0 and t.probe_cos_sim >= 0.025 for t in tops)),
        }
        dist = {
            "top_ablation_median": float(np.median([t.ablation_score for t in tops])) if tops else None,
            "top_ablation_min": float(np.min([t.ablation_score for t in tops])) if tops else None,
            "top_cos_median": float(np.median([t.probe_cos_sim for t in tops])) if tops else None,
            "top_cos_max": float(np.max([t.probe_cos_sim for t in tops])) if tops else None,
            "gap_median": (float(np.median([abs(t.ablation_score) - abs(sc.ablation_score)
                                            for t, sc in zip(tops, seconds)])) if tops else None),
        }
        result["absorption_rate"] = rates.get(1.0)
        result["absorption_rate_by_iggap"] = rates
        result["sample_portion"] = portion
        result["num_sampled"] = len(srs)
        result["funnel"] = funnel
        result["diagnostics"] = dist
        pd.DataFrame([
            {"token": s.word,
             "main_silent": all(fs.activation < EPS for fs in s.main_feature_scores),
             "top_feat": s.top_ablation_feature_scores[0].feature_id,
             "top_ablation": s.top_ablation_feature_scores[0].ablation_score,
             "top_cos": s.top_ablation_feature_scores[0].probe_cos_sim,
             "second_ablation": s.top_ablation_feature_scores[1].ablation_score}
            for s in srs
        ]).to_parquet(prop_dir / f"absorption_detail_l0{sae_info.l0}.parquet", index=False)
        print(f"[absorption] rate@1.0={rates[1.0]:.4f}; sweep {{"
              + ", ".join(f'{g}:{rates[g]:.4f}' for g in gaps) + "}}", flush=True)
        print(f"[funnel] {funnel}", flush=True)
        print(f"[diag] {dist}", flush=True)

    out = prop_dir / f"summary_l0{sae_info.l0}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"saved -> {out}\nBINARY_DONE", flush=True)


if __name__ == "__main__":
    main()
