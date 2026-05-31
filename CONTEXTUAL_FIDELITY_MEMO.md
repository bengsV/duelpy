# Fidelity Memo: Contextual Dueling Bandits vs. Source Papers

**Date:** 2026-05-31
**Scope:** Audit of the in-progress contextual scaffolding against the two reference papers.
**Verdict:** Both algorithm implementations diverge from their papers in *model* and *mechanics*,
to the point where each currently implements a different algorithm than the one it cites. The
divergences are documented below with severity ratings and concrete fixes, so the team can decide
per item whether to fix or to re-scope/rename as an "inspired-by" variant.

## Sources

- **Paper A — CoLSTIM.** Bengs, Saha & Hüllermeier, *Stochastic Contextual Dueling Bandits under
  Linear Stochastic Transitivity Models*, ICML 2022 (`bengs22a`; arXiv 2202.04593).
  Algorithm 1 ("COLSTIM").
- **Paper B — Sta'D / MaxInP.** Saha, *Optimal Algorithms for Stochastic Contextual Preference
  Bandits*, NeurIPS 2021. Algorithm 1/2 ("MaxInP"), Algorithm 3 ("Sta'D").
  *(Note: the repo's `StaD` cites this paper, but the in-tree code did not implement Algorithm 3.)*

---

## 1. The shared modelling issue (affects everything)

Both papers use the **same model family**, which the repo does **not** implement:

> A **single, shared** unknown parameter θ\* ∈ ℝᵈ (‖θ\*‖ ≤ 1). Each arm/item is represented by a
> **d-dimensional feature vector** (in Paper B these are re-supplied every round as a *context set*
> Sₜ = {xₜ¹,…,xₜᴷ}; in Paper A as a context matrix Xₜ = (xₜ,₁…xₜ,ₙ), where each xₜ,ᵢ is a *joint*
> context-and-arm feature, often a feature map). The utility of an item is the **linear score**
> θ\*ᵀx, and the pairwise preference is

> &nbsp;&nbsp;&nbsp;&nbsp; P(i ≻ j | Xₜ) = F(θ\*ᵀ(xₜ,ᵢ − xₜ,ⱼ))

> where F is a link / comparison CDF (sigmoid ⇒ BTL/logistic in Paper B; Gumbel ⇒ logistic, Gaussian
> ⇒ Thurstone, etc. in Paper A). Estimation is via the **GLM/MLE estimating equation**
> Σ (oτ − σ((xτ−yτ)ᵀθ̂))(xτ−yτ) = 0, with a **single** Gram matrix of *contrasts*
> V = Σ (xτ−yτ)(xτ−yτ)ᵀ.

### What the repo models instead

| Repo component | Repo model | Paper model | Severity |
|---|---|---|---|
| `LinearContextFeedback` | **Per-arm** θᵢ\*; one shared external context `x`; arms are fixed indices; preference = σ((θᵢ−θⱼ)ᵀx) | **Single** shared θ\*; per-arm/per-round feature vectors xₜ,ᵢ; preference = F(θ\*ᵀ(xₜ,ᵢ−xₜ,ⱼ)) | **High** — different model class |
| `ContextualPreferenceEstimate` | **Per-arm** ridge regression (Vᵢ, bᵢ, θ̂ᵢ for each arm), linear least-squares on binary y∈{0,1} | **Single** GLM/MLE on contrasts of played pairs; one Gram matrix V | **High** |
| `ArmFeatureFeedback` (Paper A) | Fixed φᵢ, shared θ\*, F=logistic/step | Closest of the three — but features are **static**, no per-round context Xₜ | **Medium** |
| `ContextualLinearRegret` | `max_i θᵢᵀx − ½(θ_iᵀx + θ_jᵀx)` | Matches paper regret notion (best score − average score of dueled pair) ✅ | **Low** (faithful) |

**Consequence:** the repo's "per-arm θ over a shared context" is a *legitimate* contextual dueling
model, but it is **neither paper's model**. Both papers deliberately use a single θ\* so that
information aggregates across arms (this is the whole point of the Õ(√(dT)) bound).

**Decision needed (#1):** Adopt the shared-θ\* / per-round-feature-set model (faithful to both
papers, and the only way the cited regret bounds hold), or keep the per-arm model and re-document it
as a distinct "arm-specific contextual" setting that does *not* claim to reproduce either paper.

---

## 2. CoLSTIM (Paper A, Algorithm 1) vs. `algorithms/colstim.py`

Paper A, Algorithm 1, per round t (after a τ-round exploration phase):

```
5:  Observe context vectors X_t = (x_{t,1} … x_{t,n})
6:  Compute MLE θ̂_t                                        # GLM MLE, eq (9)/(10)
7:  Sample B_t ~ Ber(p_t)
8-12: if B_t=1: sample ε̃_{t,i} ~ G independently per arm
      else:    sample one ε̃ ~ G, set ε̃_{t,i}=ε̃ ∀i        # coupling
13: ε_{t,i} = clip(ε̃_{t,i}, −C_thresh, C_thresh)
14: i_t = argmax_i  x_{t,i}ᵀθ̂_t + ε_{t,i}·‖x_{t,i}‖_{M_t⁻¹}    # FIRST arm: per-arm perturbed utility
15: j_t = argmax_i  ⟨z_{t,i,i_t}, θ̂_t⟩ + c₁·‖z_{t,i_t,i}‖_{M_t⁻¹}  # SECOND arm: UCB on contrast (toughest competitor)
16: M_{t+1} ← M_t + z·zᵀ ;  observe Y_t = 1[i_t ≻ j_t]      # M = Gram of contrasts
```

| # | Aspect | Paper A | Repo `CoLSTIM` | Severity |
|---|---|---|---|---|
| 2.1 | Context | Per-round Xₜ = (xₜ,₁…xₜ,ₙ) re-observed each step | **Static** φᵢ from `ArmFeatureFeedback`; no per-round context | **High** |
| 2.2 | Perturbation | **Per-arm independent scalars** εₜ,ᵢ ~ G, scaled by per-arm width ‖xₜ,ᵢ‖_{M⁻¹} | **Single Gaussian vector** θ̃ ~ 𝒩(θ̂, β²V⁻¹), then argmax φᵢᵀθ̃ | **High** — different algorithm (this is Thompson sampling, not CoLSTIM's perturb-the-index scheme) |
| 2.3 | First arm iₜ | argmax of *perturbed utility* | argmax φᵢᵀθ̃ (imitation) — same intent, wrong perturbation | **Medium** |
| 2.4 | Second arm jₜ | "Toughest competitor": argmax UCB on contrast vs iₜ (line 15) | argmax φᵢᵀθ̂ (greedy exploit); if tie with iₜ, second-best perturbed | **High** — repo logic is inverted/unrelated |
| 2.5 | Estimator | GLM **MLE** of link (eq 9/10) | Plain **ridge** on binary {0,1} | **Medium** (step link makes ridge especially off) |
| 2.6 | Gram matrix | M = Σ z·zᵀ over played contrasts (no explicit λI) | V = λI + Σ φ_diff·φ_diffᵀ — *close in spirit* ✅ | **Low** |
| 2.7 | Exploration / coupling / thresholding | τ-round init, Bernoulli coupling Bₜ, clip to ±C_thresh | None of these present | **Medium** |
| 2.8 | β schedule | c₁ width constant + perturbation params from G | `compute_beta` uses a self-normalized-style √(d·log(1+n/dλ)+2log(1/δ))+√λ | **Medium** (plausible but not the paper's constant) |
| 2.9 | Regret target | Weak regret Õ(√(dT)) under min-eigenvalue assumption | Docstring claims Õ(√(dT)) "weak regret" | claim unsupported by impl |

**Net:** the repo's CoLSTIM is a static-feature Thompson-sampling dueling bandit, not Algorithm 1.

---

## 3. Sta'D (Paper B, Algorithm 3) vs. `algorithms/stad.py`

Paper B, Algorithm 3 — stages are **nested confidence levels traversed within each round**, with
per-stage "informative sample" sets φˢ and a single shared MLE:

```
init: t₀ random pairs; V_{t0+1} = Σ contrasts; φˢ ← [t₀] ∀s
each round t (start s=1, G¹=[K]); repeat:
  8:  MLE θ̂ˢ on the samples in φˢ (GLM estimating equation)
  9:  Vₜˢ = Σ_{τ∈φˢ} (xτ−yτ)(xτ−yτ)ᵀ
  10: gₜˢ(i) = θ̂ˢᵀxₜⁱ ;  pₜˢ(i,j) = η·‖xₜⁱ−xₜʲ‖_{(Vˢ)⁻¹}
  11: if all pairs pₜˢ ≤ 1/√T:   aₜ=argmax g; bₜ=argmax (g(b)+p(b,aₜ)); PLAY (exploit)
  15: elif all pairs pₜˢ ≤ 1/2ˢ:  Bˢ={i: ∃j, g(i)+1/2ˢ < g(j)}; G^{s+1}=Gˢ\Bˢ; s←s+1  (eliminate)
  18: else:                       pick any pair with p>1/2ˢ; φˢ←φˢ∪{t}; PLAY (explore)
```

| # | Aspect | Paper B (Alg 3) | Repo `StaD` | Severity |
|---|---|---|---|---|
| 3.1 | Parameter | **Single** shared θ\* via GLM MLE | **Per-arm** ridge θ̂ᵢ (`ContextualPreferenceEstimate`) | **High** |
| 3.2 | Context | Per-round **feature set** Sₜ={xₜ¹…xₜᴷ} | Single external `x`, fixed arm indices | **High** |
| 3.3 | Stage meaning | Nested confidence levels (1/2ˢ schedule) traversed **within one round** | **Blocks of rounds** of doubling length (2^stage) | **High** — different control flow |
| 3.4 | Sample tracking | Per-stage independent sets φˢ ("stagewise sample independence", Lemma 14) — the crux of the Õ(√(dT)) bound | No φˢ; all samples pooled | **High** |
| 3.5 | Pair selection | exploit/eliminate/explore branch on pₜˢ thresholds (1/√T, 1/2ˢ) | top-2 UCB arms from active set | **High** |
| 3.6 | Elimination | g(i)+1/2ˢ < g(j) on current context | UCBᵢ < max LCBⱼ at end of block | **High** |
| 3.7 | Estimator | logistic GLM MLE | linear ridge on binary y | **Medium** |
| 3.8 | Docstring | — | Self-describes as "approximate implementation" | honest, but it's a *different* algorithm, not an approximation |

**Net:** the repo's `StaD` is a per-arm UCB elimination scheme over a shared context; it shares only
the name and the "stagewise" intuition with Algorithm 3. `MaxInP` (Paper B's simpler Õ(d√T)
algorithm) is **not implemented at all** and is the easier, higher-value first target.

---

## 4. What is already correct / reusable

- `ContextualLinearRegret` matches the papers' regret notion (best score − average of dueled pair). ✅
  (Minor: paper B sometimes uses g=σ(θᵀx); the linear-score version here is fine and standard.)
- `LinearContextEnvironment`'s separation of arm-parameter RNG vs. context RNG is good hygiene. ✅
- The contrast-Gram-matrix update in `CoLSTIM` (V += φ_diff·φ_diffᵀ) is structurally the paper's Mₜ. ✅
- The abstract `ContextualAlgorithm` run-loop (context_generator) is a reasonable host for Paper B —
  but `context_generator` must yield a **set of K feature vectors per round**, not a single vector.

---

## 5. Recommended remediation (in priority order)

1. **Lock the model (Decision #1).** Recommended: adopt the shared-θ\* / per-round-feature-set model.
   This is a rewrite of `LinearContextFeedback` (→ yield Sₜ of K feature vectors) and
   `ContextualPreferenceEstimate` (→ single GLM MLE on contrasts, one Gram matrix V).
2. **Implement `MaxInP` first** (Paper B, Alg 2) — it is small, uses exactly the shared MLE +
   contrast-Gram machinery, gives a clean Õ(d√T) baseline, and de-risks the model rewrite.
3. **Reimplement `StaD` faithfully** (Paper B, Alg 3) — nested-stage control flow, φˢ sample sets,
   1/2ˢ and 1/√T thresholds. Or, if a faithful port is out of scope, **rename** the current class
   (e.g. `ContextualEliminationUCB`) and drop the NeurIPS-2021 attribution.
4. **Reimplement `CoLSTIM` faithfully** (Paper A, Alg 1) — per-round Xₜ, per-arm scalar perturbations
   from a pluggable distribution G, τ-exploration + Bernoulli coupling + thresholding, GLM MLE, and
   the two-arm rule (lines 14–15). Or rename to reflect the Thompson-sampling variant actually built.
5. **MLE solver.** Add a small GLM/logistic MLE (Newton or scipy) to `stats/`, shared by all four.
   This is the single biggest missing primitive.
6. **Tests.** Current tests assert "runs and returns a winner"; add tests that (a) recover a known θ\*
   from synthetic logistic data and (b) check sublinear regret growth on `LinearContextEnvironment`.

## 6. Open questions for the team

- **Q1 (model):** shared θ\* (faithful) vs. per-arm θᵢ (current)? — gates items 1–4.
- **Q2 (link):** support a pluggable F (logistic/Gaussian/step) as both papers do, or fix logistic?
- **Q3 (scope):** faithful reproductions, or keep the current variants and just re-attribute/rename?
- **Q4 (MaxInP):** add it? It's the natural Õ(d√T) baseline and the cheapest faithful win.
