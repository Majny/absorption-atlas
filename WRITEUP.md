# Does the SAE-absorption metric over-report for non-spelling features?

*A short research note. Jakub Dvořák. Gemma-2-2b/9b, Gemma Scope SAEs.*

**TL;DR (the one claim, stated up front).** SAE "feature absorption" (Chanin et al., 2024) is measured
in the literature almost entirely on the *first-letter spelling* task. I run the two standard absorption
metrics, the **projection** metric (the current SAEBench standard) and the older **ablation/behavioral**
metric, on a structurally different property, **is-capitalized**. The two metrics *disagree*: the
projection metric reports single-dominant-latent absorption modestly above its own random-direction null
(0.12 vs 0.03, R=3), while the ablation metric finds almost no absorbed tokens: **1 of 198** scored
main-silent candidates on Gemma-2-2b and **1 of 35** on Gemma-2-9b, at the same 3× dominance ratio (versus
0.41 and 0.43 for spelling). Two controls make the boring explanations unlikely: it is not the
model failing the task (9b does is-capitalized at 0.91 accuracy, still 1/35), and not a dead layer (at
the same 9b layer/L0, the ablation metric fires at 0.045 for spelling). Per probe true positive under the
same 3× ratio, spelling scores 0.038 (2b) and 0.049 (9b) against ~0.0001 for is-capitalized. So: the field-standard projection metric
appears to **over-report** single-latent absorption for a distributed structural feature that has almost no
single-latent causal correlate. The effect is small and rests on one structural property; its limits are spelled
out below.

## Background

An SAE latent that looks monosemantic for a concept ("starts-with-S") can silently fail on some tokens
while a token-aligned latent ("short") carries the concept instead: *feature absorption* (Chanin et al.,
[2409.14507](https://arxiv.org/abs/2409.14507)). Two metrics quantify it:
- **ablation / behavioral:** integrated-gradient attribution of SAE latents to the model's *task answer*
  (e.g. the first-letter logit). Chanin's original; known to **decay to ~0 in later layers**, which is
  exactly why SAEBench replaced it.
- **projection / representational:** decompose the residual's projection onto the concept's probe
  direction into per-latent contributions; a single dominant non-main latent = absorption. This **is**
  the SAEBench-standard absorption metric ([2503.09532](https://arxiv.org/abs/2503.09532)).

Chanin's Future Work explicitly asks for "absorption unrelated to character identification." Feature
Hedging (Chanin, [2505.11756](https://arxiv.org/abs/2505.11756)) already k-sparse-probes part-of-speech,
so structural properties are not virgin territory. But running the absorption metrics themselves
(both) on structural properties, and contrasting them, has not been done to my knowledge.

## What I did (and reproduced)

- Reproduced first-letter absorption on current Gemma Scope SAEs (Gemma-2-2b, L12/16k): mean rate 0.032,
  and the **L0 law** (absorption rises as L0 falls: L0 22→0.127 … 445→0.010), independently consistent
  with SAEBench. (Reproduction, not new.)
- Extended to structural properties (is-capitalized, all-caps, suffix -ing/-ed). is-capitalized (727 candidates
  on 2b, 273 on 9b; 130–198 scored per metric on 2b, 35 on 9b) is the only well-powered anchor; all-caps/suffix (n=9–33) are underpowered anecdotes.

## The result: the two metrics disagree for is-capitalized

**Projection (SAEBench standard), single-dominant-latent rate at R=3, Wilson 95% CI over the
main-latent-silent candidate pool:**

| property | projection | random-direction null |
|---|---|---|
| first-letter | **0.53** [0.51, 0.55] (n=2085) | 0.04 [0.03, 0.05] |
| is-capitalized | **0.12** [0.077, 0.191] (n=130) | 0.03 [0.014, 0.081] |

So projection reports absorption for is-capitalized above the null's point estimate, but only **marginally** (the Wilson
intervals overlap slightly at n=130: [0.077, 0.191] vs [0.014, 0.081]) and ~4× weaker than spelling.

**Ablation (behavioral):** Chanin's criterion requires the top attribution to beat the runner-up by an
*absolute* IG gap (≥1.0, calibrated to the 26-way letter logit). On the binary True/False logit the largest
gap among scored is-capitalized tokens is 0.050 (2b) and 0.146 (9b), below every threshold tested
(0.5–4.0), so its 0.0 is forced by scale. With a scale-free ratio instead (main latent silent, top
attribution negative and probe-aligned, |top| ≥ R·|second|), conditioned on main-silent scored tokens:

| ablation, conditioned | R=1.5 | R=2 | R=3 |
|---|---|---|---|
| first-letter, 2b (n=2283) | 0.64 [0.62, 0.66] | 0.54 [0.52, 0.56] | 0.41 [0.39, 0.43] |
| is-capitalized, 2b (n=198) | 0.056 [0.031, 0.097] | 0.010 [0.003, 0.036] | 0.005 [0.001, 0.028] |
| first-letter, 9b (n=2901) | 0.66 [0.65, 0.68] | 0.58 [0.56, 0.59] | 0.43 [0.41, 0.45] |
| is-capitalized, 9b (n=35) | 0.17 [0.08, 0.33] | 0.086 [0.03, 0.22] | 0.029 [0.005, 0.145] |

So on spelling the two metrics roughly agree (ablation 0.41 vs projection 0.53 at R=3), while on
is-capitalized ablation is near zero (1/198 on 2b, 1/35 on 9b at R=3).

(Coverage and denominators, stated precisely: on 2b both metrics run over the same 200-token sample of
the 727-candidate pool. The ablation calculator first filters to prompts whose main latents are silent
(308 of 727 qualify; an active main latent precludes absorption by definition) and IG-scores a random
200 of those; the projection rate additionally conditions per token on *full* main-latent silence,
leaving n=130, so the two conditioned pools overlap but are not identical tokens. On 9b every candidate
is accounted for: 238 of 273 have an active main latent and the remaining 35 were all scored.)

## Controls: it is not task-validity and not a dead layer

| condition | task accuracy | behavioral absorption (per probe TP, absolute gap) | conditioned, R=3 |
|---|---|---|---|
| first-letter (2b, L12) | ~0.95 | 0.032 | 0.41 |
| first-letter (9b, L20) | ~0.96 | **0.045** | 0.43 |
| is-capitalized (2b, L12) | 0.59 | 0.0 (forced by scale) | 0.005 (1/198) |
| **is-capitalized (9b, L20)** | **0.91** | **0.0** (forced by scale) | **0.029 (1/35)** |

- **Not task-validity:** 9b performs is-capitalized at 0.91 (vs 2b 0.59), yet behavioral absorption stays near zero (1/35).
- **Not a dead layer:** at the same 9b layer/L0, spelling behavioral absorption is 0.045 (0.43 conditioned).

**Side result (a genuine confound for the ablation metric):** *within* first-letter, behavioral
absorption tracks task accuracy: sweeping accuracy 0.53→0.97 (via ICL count and corrupted ICL) moves
behavioral absorption 0.011→0.030, while the projection metric (at R=2 here) stays roughly flat at 0.55–0.62. So the ablation metric
is task-performance-sensitive; the projection metric is not.

## What it means, and the alternative I can't fully rule out

The defensible reading is a **metric-validity** one: for a distributed structural feature, the projection
metric's "single dominant latent" criterion reads modestly above chance, but there is **almost no causal
single-latent correlate**. The most likely mundane mechanism is **feature hedging** (Chanin 2505.11756):
capitalization is a high-frequency, correlated concept that a 16k SAE spreads across several latents, so
"one dominant latent by projection" is a weak, partly-geometric signal rather than a token-aligned
absorber that the model actually *uses*. I do not claim a deep "representational-vs-causal
dissociation"; I claim the narrower, checkable thing: the two metrics disagree here, projection over-reports
relative to ablation, and the disagreement survives task-ability and layer controls.

## Limitations

- **Ablation threshold scale.** Chanin's absolute IG-gap threshold cannot fire on the binary logit, so
  the is-capitalized ablation numbers rest on the scale-free ratio R, a substitute introduced for this reason (`analyze_scalefree.py`).
  The 9b ablation pool is small (n=35, R=3 CI up to 0.145).
- **Near-floor effects.** Spelling behavioral absorption is only ~0.01–0.05 of probe true positives anywhere; the projection
  is-capitalized signal (0.12) is marginal above its null. Small numbers.
- **One structural property.** is-capitalized is the only well-powered one; all-caps/suffix are n=9–33.
  "Property-type-dependent" is really n=1 until more properties are powered up.
- **One model family, one SAE recipe.** 2b + 9b are cross-*scale* within Gemma-2; only Gemma Scope /
  JumpReLU SAEs. No Matryoshka/TopK, no non-Gemma model.
- **No downstream "so what."** I don't show a use-case that trusting the projection-flagged absorption
  would break.
- **SAE loading.** SAEs were loaded with sae-lens<4, whose `fold_W_dec_norm` leaves JumpReLU thresholds
  unscaled; all six Gemma Scope SAEs used have unit-norm decoder rows, so this is a no-op here.

## What would make this a workshop paper (not done here)

More structural properties at n>130; one non-Gemma or non-JumpReLU point; and one downstream demonstration.

*Code + all figures/numbers: [github.com/Majny/absorption-atlas](https://github.com/Majny/absorption-atlas);
full 15-page technical report with appendices: [paper/paper.pdf](https://github.com/Majny/absorption-atlas/blob/main/paper/paper.pdf).
The committed results cover most numbers above. The ablation detail parquets for first-letter
(`results/gate/`, `results/first_letter_9b/`) and 9b is-capitalized (`results/is_capitalized_9b/`) are not
committed; their counts, with each parquet's sha256, are in
[`results/absorption_counts.json`](results/absorption_counts.json) (written by `count_absorption.py`).
The raw task-accuracy outputs are not committed either (the accuracies are in the paper's tables).*

*Engineering and drafting were AI-assisted (Claude); the research direction, experimental design,
verification of every number, and the conclusions are mine.*
