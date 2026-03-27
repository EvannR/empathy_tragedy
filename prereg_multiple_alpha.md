# Preregistration: Dose-Response Relationship Between Empathy Weight (Alpha) and Collective Cooperation in a Multi-Agent Tragedy of the Commons

## 1. Title, Authors, and Context

**Title**: Dose-response relationship between empathy weight (alpha) and collective cooperation in a multi-agent tragedy of the commons

**Authors**: Evann, Jerome

**Date**: 2026-03-16

**Data status**: Data have already been collected. Simulations were run prior to the writing of this preregistration. This document is written to formalize hypotheses and analysis plans transparently, acknowledging that it constitutes a post-data preregistration.

**Prior knowledge**: Results from Experiment 1 (binary empathic vs. non-empathic) showed that empathic agents (alpha=0.5, see_emotions=True) outperformed non-empathic agents (alpha=0.0, see_emotions=False) on resource preservation. This experiment extends the investigation to a parametric sweep of alpha values.

---

## 2. Hypotheses

### Primary hypotheses

**H1 — Monotonic resource preservation**: Increasing alpha monotonically improves resource preservation. Specifically, higher alpha values lead to:
- (H1a) Lower final resource depletion
- (H1b) Lower cumulative resource depletion
- (H1c) Longer episode duration (more steps before resource exhaustion)

Operationalization: A significant positive linear trend between alpha and each DV, tested via Jonckheere-Terpstra trend test or linear regression.

**H2 — Monotonic inequality reduction**: Increasing alpha monotonically reduces reward inequality across agents, as measured by a lower Gini coefficient on total personal rewards.

Operationalization: A significant negative linear trend between alpha and Gini coefficient.

### Exploratory hypotheses

**H3 — Nonlinear threshold effect**: The relationship between alpha and cooperation is nonlinear. There exists a critical alpha value beyond which cooperation sharply increases (phase transition).

Operationalization: Visual inspection of dose-response curves; segmented regression to detect a breakpoint.

**H4 — Diminishing returns at high alpha**: Marginal gains in resource preservation decrease at high alpha values (0.85, 0.99), suggesting diminishing returns to empathy.

Operationalization: Comparison of marginal gains between consecutive alpha levels at the high end of the range.

---

## 3. Variables

### Independent variable

- **Alpha (empathy weight)**: 7 levels — {0.0, 0.15, 0.25, 0.5, 0.75, 0.85, 0.99}
  - All conditions have `see_emotions = True` (agents observe others' average emotion)
  - Alpha determines the weighting of others' well-being in each agent's reward:
    `combined_reward = (1 - alpha) * personal_satisfaction + alpha * empathic_reward`

### Primary dependent variables

1. **Resource depletion (final)**: `1 - (resource_remaining / 500)` at episode end. Range: [0, 1]. Higher = worse preservation.

2. **Resource depletion (cumulative)**: `1 - (mean_resource_across_steps / 500)`. Captures resource trajectory, not just endpoint.

3. **Episode length**: `total_steps` per episode. Range: [1, 1000]. Longer = better preservation (resource lasts longer).

4. **Gini coefficient**: Computed from the 6 agents' total personal rewards per episode. Range: [0, 1]. Higher = more unequal distribution.

### Exploratory dependent variables

5. **Cooperation rate**: Fraction of abstain actions (action=0) across all agents and steps per episode.

6. **Per-agent efficiency**: Mean personal reward per agent per episode.

### Fixed parameters (not varied)

| Parameter | Value | Description |
| --- | --- | --- |
| see_emotions | True | Agents observe average emotion of others |
| beta | 0.5 | Weight of last meal vs. history in personal satisfaction |
| smoothing | linear | Emotion mapping function |
| threshold | 0.5 | Neutral emotion threshold |
| emotion_decimals | 2 | Rounding precision for emotion signal |
| emotion_type | average | Agents observe mean emotion (scalar) |

---

## 4. Experimental Design and Paradigm

### Study type

Computational simulation experiment (between-condition comparison across alpha levels).

### Environment: Tragedy of the Commons

- **Agents**: 6 DQN agents sharing a common resource pool
- **Initial resources**: 500 units
- **Resource dynamics**: Stochastic exploitation — success probability = `resource / 500`. Each successful exploitation removes 1 unit.
- **Regeneration**: `new_resource = max(0, (resource - consumed) * regen_rate)` with regen_rate = 1.0
- **Actions**: Binary — exploit (1) or abstain (0)
- **Episode termination**: Resource reaches 0 or max steps (1000) reached
- **Episodes per run**: 500
- **Homogeneous alpha**: All 6 agents within a run share the same alpha value

### Agent architecture: Deep Q-Network (DQN)

- Input: average emotion of other agents (1-dimensional)
- Hidden layers: 2 layers, 64 neurons each, ReLU activation
- Output: Q-values for 2 actions (abstain, exploit)
- Experience replay buffer
- Target network updated every 5 steps
- Hyperparameters: learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, batch_size=16

### Emotion and reward calculation

**Emotion**: Each agent tracks a sliding window of 10 recent meals (binary). Consumption rate = sum(meals) / 10. Emotion is linearly mapped to [-1, 1] based on threshold (0.5):
- rate >= 0.5: emotion = (rate - 0.5) / 0.5
- rate < 0.5: emotion = -(0.5 - rate) / 0.5

**Personal satisfaction**: `beta * last_meal + (1 - beta) * history_rate` (beta=0.5)

**Empathic reward**: Mean emotion of the other 5 agents

**Combined reward**: `(1 - alpha) * personal + alpha * empathic`

---

## 5. Sample

- **Conditions**: 7 alpha levels
- **Runs per condition**: 3 (independent random seeds)
- **Total simulations**: 7 x 3 = 21
- **Episodes per run**: 500
- **Total episode-level observations**: 10,500

**Sample size justification**: 3 runs per condition is consistent with prior experiments in this project (Experiments 1 and 2). The primary analysis aggregates over 500 episodes per run, providing 1,500 episode-level data points per alpha level. This sample size is constrained by computational cost (each simulation involves 6 DQN agents learning over 500 episodes of up to 1,000 steps each).

**Seeding**: Each run uses a deterministic seed via `set_global_seed()`, fixing numpy, random, and torch for reproducibility.

---

## 6. Statistical Analysis Plan

### Unit of analysis

Episode-level DVs aggregated per condition. Where run-level aggregation is needed (e.g., for ANOVA with N=3 per condition), each run's mean across 500 episodes serves as the observation.

### Tests for H1 (monotonic resource preservation)

For each DV (final depletion, cumulative depletion, episode length):

1. **Jonckheere-Terpstra trend test** for ordered alternatives (one-sided, testing monotonic trend across 7 alpha levels). This is the primary confirmatory test.
2. **Linear regression**: DV ~ alpha, to estimate the slope and R-squared of the dose-response relationship.
3. **One-way Kruskal-Wallis test** (7 groups, N=3 per group at run level) as a supplementary omnibus test.
4. **Post-hoc pairwise comparisons**: Mann-Whitney U tests between each alpha level and the baseline (alpha=0.0), with Bonferroni correction (6 comparisons per DV).

### Tests for H2 (monotonic inequality reduction)

Same approach as H1, applied to the Gini coefficient (expected negative trend).

### Tests for H3 (exploratory: threshold effect)

- Visual inspection of mean DV vs. alpha dose-response curves with error bars (min-max or SD across 3 runs).
- Segmented regression (piecewise linear) to detect a breakpoint in the dose-response curve.
- No multiple comparison correction applied (exploratory).

### Tests for H4 (exploratory: diminishing returns)

- Compute marginal gain: DV(alpha_k+1) - DV(alpha_k) for consecutive alpha levels.
- Visual comparison of marginal gains at high vs. low alpha.
- No formal test pre-specified (exploratory).

### Learning trajectory analysis

- Windowed moving averages of combined reward per episode (window size to be determined, e.g., 50 episodes) to visualize learning curves per alpha level.
- Mean and min-max envelope across 3 runs.

### Multiple comparison corrections

- **Primary analyses (H1, H2)**: Bonferroni correction applied across the 3 primary DVs (alpha_corrected = 0.05 / 3 = 0.0167) for omnibus tests. Post-hoc pairwise tests corrected within each DV (Bonferroni for 6 comparisons: alpha_corrected = 0.05 / 6 = 0.0083).
- **Exploratory analyses (H3, H4)**: No correction applied; results reported as exploratory.

### Significance threshold

Alpha = 0.05 (before correction).

### Software

- Python 3.x
- scipy (Kruskal-Wallis, Mann-Whitney)
- statsmodels (regression, trend tests)
- matplotlib / seaborn (visualization)
- numpy, pandas (data manipulation)

### Missing data

Simulations are deterministic given a seed and run to completion. No missing data is expected. If a simulation file is missing or corrupted, it will be excluded and reported.

### Outlier handling

All 500 episodes per run are included in the analysis. No episodes are excluded (early exploration episodes are part of the learning process and contribute to the trajectory analysis).

---

## 7. Exploratory Analyses

The following analyses are explicitly labeled as exploratory and will not be used for confirmatory inference:

1. **Cooperation rate**: Fraction of abstain actions across alpha levels — does restraint increase with empathy?
2. **Per-agent learning curves**: Windowed moving averages per agent to assess whether individual agents converge to different strategies.
3. **Resource depletion trajectories**: Episode-by-episode resource remaining curves per alpha level.
4. **Within-run agent heterogeneity**: Variance in personal rewards across 6 agents within a run — do agents specialize into different roles?
5. **Nonlinear model fitting**: Fit sigmoid or logistic curves to the dose-response relationship to characterize the functional form.
6. **Comparison with Experiment 1**: Qualitative comparison of alpha=0.0 and alpha=0.5 results with the original binary experiment.
