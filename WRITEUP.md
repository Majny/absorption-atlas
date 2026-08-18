# Does the SAE-absorption metric over-report for non-spelling features?

*A metric-validity note. Jakub Dvořák, 2026-07. Gemma-2-2b/9b, Gemma Scope SAEs.*

**TL;DR (the one claim, stated up front).** SAE "feature absorption" (Chanin et al., 2024) is measured
in the literature almost entirely on the *first-letter spelling* task. I run the two standard absorption
metrics — the **projection** metric (the current SAEBench standard) and the older **ablation/behavioral**
metric — on a structurally different property, **is-capitalized**. The two metrics *disagree*: the
projection metric reports single-dominant-latent absorption modestly above its own random-direction null
(0.12 vs 0.03, R=3), while the ablation metric reports exactly zero on every candidate token:
**0/727** on Gemma-2-2b and **0/273** on Gemma-2-9b. Two controls rule out the boring explanations: it is not the
model failing the task (9b does is-capitalized at 0.91 accuracy, still 0.0), and not a dead layer (at
the same 9b layer/L0, the ablation metric fires at 0.045 for spelling; at matched task accuracy ~0.9 it
fires at ~0.017–0.035 for spelling but 0.0 for is-capitalized). So: the field-standard projection metric
appears to **over-report** single-latent absorption for a distributed structural feature that has no
causal correlate. The effect is small and rests on one structural property; its limits are spelled
out below.

## Background

An SAE latent that looks monosemantic for a concept ("starts-with-S") can silently fail on some tokens
while a token-aligned latent ("short") carries the concept instead — *feature absorption* (Chanin et al.,
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
(both) on structural properties, and contrasting them, has not been done.

## What I did (and reproduced)

- Reproduced first-letter absorption on current Gemma Scope SAEs (Gemma-2-2b, L12/16k): mean rate 0.032,
  and the **L0 law** (absorption rises as L0 falls: L0 22→0.127 … 445→0.010), independently consistent
  with SAEBench. (Reproduction, not new.)
- Extended to structural properties (is-capitalized, all-caps, suffix -ing/-ed). is-capitalized (n≈130–273
  candidates) is the only well-powered anchor; all-caps/suffix (n=9–33) are underpowered anecdotes.

## The result: the two metrics disagree for is-capitalized

**Projection (SAEBench standard), single-dominant-latent rate at R=3, Wilson 95% CI over the
main-latent-silent candidate pool:**

| property | projection | random-direction null |
|---|---|---|
| first-letter | **0.53** [0.51, 0.55] (n=2085) | 0.04 [0.03, 0.05] |
| is-capitalized | **0.12** [0.077, 0.191] (n=130) | 0.03 [0.014, 0.081] |

So projection reports above-null absorption for is-capitalized — but **marginally** (the CIs barely
separate at n=130) and ~4× weaker than spelling.

**Ablation (behavioral):** is-capitalized = **0.0** on Gemma-2-2b *and* Gemma-2-9b. On 9b, **0 of 273**
candidate tokens absorbed (binomial 95% CI [0, ~0.011]). A robust zero.

(The two metrics see different denominators by construction: the ablation calculator evaluates every
candidate token from the absorption pipeline — 727 on 2b — while the projection rate is conditioned on
the subset of candidates whose main latent is *fully silent*, n=130 on 2b.)

## Controls: it is not task-validity and not a dead layer

| condition | task accuracy | behavioral absorption |
|---|---|---|
| first-letter (2b, L12) | ~0.95 | 0.032 |
| first-letter (9b, L20) | ~0.96 | **0.045** |
| is-capitalized (2b, L12) | 0.59 | 0.0 |
| **is-capitalized (9b, L20)** | **0.91** | **0.0** |

- **Not task-validity:** 9b performs is-capitalized at 0.91 (vs 2b 0.59), yet behavioral absorption stays 0.
- **Not a dead layer:** at the same 9b layer/L0, spelling behavioral absorption is 0.045.
- **Matched accuracy:** using an ICL-count + corrupted-ICL sweep on 2b first-letter, behavioral absorption
  at accuracy ~0.9 is ~0.017–0.035 — so at *matched* task ability, spelling absorbs and is-capitalized
  does not (0.0). This is the airtight version of "not task-validity."

**Side result (a genuine confound for the ablation metric):** *within* first-letter, behavioral
absorption tracks task accuracy — sweeping accuracy 0.53→0.97 (via ICL count and corrupted ICL) moves
behavioral absorption 0.011→0.030, while the projection metric stays flat ~0.60. So the ablation metric
is task-performance-sensitive; the projection metric is not.

## What it means, and the alternative I can't fully rule out

The defensible reading is a **metric-validity** one: for a distributed structural feature, the projection
metric's "single dominant latent" criterion reads modestly above chance, but there is **no causal
single-latent correlate**. The most likely mundane mechanism is **feature hedging** (Chanin 2505.11756):
capitalization is a high-frequency, correlated concept that a 16k SAE spreads across several latents, so
"one dominant latent by projection" is a weak, partly-geometric signal rather than a token-aligned
absorber that the model actually *uses*. I do not claim a deep "representational-vs-causal
dissociation"; I claim the narrower, checkable thing: the two metrics disagree here, projection over-reports
relative to ablation, and the disagreement survives task-ability and layer controls.

## Limitations

- **Near-floor effects.** Spelling behavioral absorption is only ~0.01–0.05 anywhere; the projection
  is-capitalized signal (0.12) is marginal above its null. Small numbers.
- **One structural property.** is-capitalized is the only well-powered one; all-caps/suffix are n=9–33.
  "Property-type-dependent" is really n=1 until more properties are powered up.
- **One model family, one SAE recipe.** 2b + 9b are cross-*scale* within Gemma-2; only Gemma Scope /
  JumpReLU SAEs. No Matryoshka/TopK, no non-Gemma model.
- **No downstream "so what."** I don't show a use-case that trusting the projection-flagged absorption
  would break.

## What would make this a workshop paper (not done here)

More structural properties at n>130; one non-Gemma or non-JumpReLU point; and one downstream demonstration.
Given a Dec-2026 timeline, this note + the reproduction is the artifact as-is; the archival version
is future work.

*Code + all figures/numbers: [github.com/Majny/absorption-atlas](https://github.com/Majny/absorption-atlas);
full 15-page technical report with appendices: [paper/paper.pdf](https://github.com/Majny/absorption-atlas/blob/main/paper/paper.pdf).
Every claim above is reproducible from the committed results.*

*Engineering and drafting were AI-assisted (Claude); the research direction, experimental design,
verification of every number, and the conclusions are mine.*
