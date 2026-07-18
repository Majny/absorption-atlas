# Absorption Atlas

**Does SAE feature absorption generalize beyond first-letter spelling?**

Sparse autoencoders (SAEs) are meant to decompose a language model's activations into
reliable, monosemantic features. **Feature absorption** (Chanin et al.,
[2409.14507](https://arxiv.org/abs/2409.14507)) is a known failure mode: a specific latent
"absorbs" the direction of a more general one, so the general latent silently fails to fire —
e.g. a *starts-with-L* latent stays dark on *lion* because a dedicated *lion* latent swallowed
the direction. This undermines the SAE-as-decomposition premise.

Today absorption is measured on essentially **one task**: "token starts with letter X". Whether
absorption is a **general** SAE failure mode or a **quirk of alphabetical spelling** is open.
This project measures absorption across a battery of other objective, tokenizer-derivable token
properties (last-letter, is-capitalized, ALL-CAPS, is-numeric, common suffixes) on off-the-shelf
public SAEs (Gemma Scope), to answer that question.

## Status

Early / in progress. **Week 1 gate:** reproduce the published first-letter absorption on a
current Gemma Scope SAE before extending. See [`PLAN.md`](PLAN.md).

## Method (from Chanin et al. + their reference code)

For a concept, three stages (all from the [`sae-spelling`](https://github.com/lasr-spelling/sae-spelling) repo):

1. **LR probe** on the residual stream → the ground-truth concept direction.
2. **k-sparse probing** → the "main"/split SAE latents that predict the concept.
3. **IG-ablation absorption calculator** → on tokens where the probe fires but the main latents
   don't, integrated-gradient attribution confirms a single, probe-aligned latent causally
   carries the concept. `absorption_rate = #absorbed / #probe-true-positives`, per concept.

Extending to a new property needs its own ICL formatter + logit-diff `metric_fn` + probe +
k-sparse results; the calculator and aggregation are reused. Key gotchas (recalibrate the
IG-gap threshold per property; verify a clean "main" latent even exists via the k-sparse F1
curve; layer ceiling ≤17 for Gemma-2-2b) are documented in `PLAN.md`.

## Setup

```bash
# 1. clone the upstream methodology repo alongside this one (gitignored here)
git clone --depth 1 https://github.com/lasr-spelling/sae-spelling.git
cd sae-spelling && uv venv --python 3.11 .venv && uv pip install --python .venv -e . && cd ..

# 2. Gemma-2-2b is HF-gated: accept the license at huggingface.co/google/gemma-2-2b,
#    then put a read token in a gitignored .env:  HF_TOKEN=hf_...
```

## Run the gate

```bash
cd sae-spelling && .venv/bin/python ../run_gate.py --layer 12 --width 16000 --max-prompts 200
```

Produces `results/gate/gate_summary.json` with the per-letter and mean absorption rate.

## Prior art

- Chanin et al., *A is for Absorption* ([2409.14507](https://arxiv.org/abs/2409.14507)) — defines
  + measures absorption, first-letter only.
- SAEBench ([2503.09532](https://arxiv.org/abs/2503.09532)) — standardizes the metric, first-letter only.

## License / credit

Builds directly on the `sae-spelling` implementation by the LASR spelling team. This repo adds
the multi-property extension + replication.
