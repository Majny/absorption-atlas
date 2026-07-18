# Gate verification — is the reproduction faithful?

Independent literature check of the Week-1 gate result (mean full-absorption rate **0.032** on
Gemma-2-2b, Gemma Scope **layer 12 / 16k / L0 ≈ 82**, first-letter task, 200 prompts/letter).

**Verdict: CONSISTENT** — within the plausible published range, right L0-dependence, right
per-letter shape, right metric. Not suspiciously off.

## Metric identity (avoid the apples-to-oranges trap)
Our `sae-spelling` metric = SAEBench's **`mean_full_absorption_score`** (Chanin's strict
full-absorption: probe fires, main latents silent, one probe-aligned latent causally carries the
direction). Do **not** compare to SAEBench's newer lenient `mean_absorption_fraction_score`.

## Reference: SAEBench `mean_full_absorption_score` vs L0 (JumpReLU, gemma-2-2b, L12, 16k)
Raw results from `huggingface.co/datasets/adamkarvonen/sae_bench_results_0125`:

| L0 | full_absorption |
|----|-----------------|
| 21.3  | 0.364 |
| 42.5  | 0.256 |
| 85.3  | **0.109** |
| 172.0 | 0.064 |
| 353.7 | 0.027 |
| 711.3 | 0.014 |

Absorption rises steeply as L0 falls — matches Chanin's law ("wider & lower-L0 SAEs absorb more").
Our L0≈82 lands on the L0=85 row.

## Where 0.032 sits
- SAEBench's own-suite JumpReLU at L0≈85 = **0.109** → our 0.032 is ~3.4× lower.
- BUT that anchor is SAEBench's *quickly-trained* JumpReLU, not Google's **Gemma Scope** (which we
  used). Gemma Scope is trained more thoroughly and consistently shows **lower** absorption at
  matched L0. Independent JumpReLU points: Adaptive Temporal Masking (arXiv:2510.08855, Table 1)
  reports gemma-2-2b L12/16k JumpReLU absorption **0.0114**; a secondary config ~0.0167.
- Plausible published range for this config ≈ **0.011–0.36** (L0-dominated). **0.032 sits
  comfortably inside**, between the well-trained-JumpReLU floor and the SAEBench suite anchor. A
  faithful Gemma Scope reproduction landing *below* SAEBench's weaker suite is the expected
  direction, not a red flag.

## Per-letter pattern
`s` among the highest = exactly Chanin's worked example ("starts-with-S" absorbed by a "short"
latent, §5.2). High per-letter variance with a few dominant letters = Chanin Fig 17's qualitative
picture. (The specific u/i/c ranking is not checkable against a published numeric per-letter table.)

## Residual uncertainty / definitive check
The exact SAEBench score for the *specific* Gemma Scope L0≈82 checkpoint isn't in the 0125 results
dataset (only SAEBench's own suite + baselines are). The definitive apples-to-apples number lives on
**Neuronpedia's SAE Bench dashboard**. Two stronger confirmations pursued here:
1. **L0 sweep** (this repo, layer 12/16k, L0 ∈ {22,41,82,176,445}) — reproduce the *shape* of the
   curve above on the actual Gemma Scope SAEs (should decrease steeply with L0, sitting somewhat
   below the SAEBench JumpReLU line).
2. Read the Gemma Scope L12/16k absorption off Neuronpedia for the exact number.

Sources: Chanin et al. arXiv:2409.14507 (§5.2–5.3, Figs 7b/17); SAEBench arXiv:2503.09532;
`adamkarvonen/sae_bench_results_0125`; Adaptive Temporal Masking arXiv:2510.08855 Table 1.
