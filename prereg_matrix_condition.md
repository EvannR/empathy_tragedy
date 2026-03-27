# Preregistration: Disentangling Observation and Reward Components of Empathy in Multi-Agent Cooperation — A 2x2 Factorial Design

## 1. Title, Authors, and Context

**Title**: Disentangling observation and reward components of empathy in multi-agent cooperation: A 2x2 factorial design

**Authors**: Evann, Jerome

**Date**: 2026-03-16

**Data status**: Data have already been collected. Simulations were run prior to the writing of this preregistration. This document is written to formalize hypotheses and analysis plans transparently, acknowledging that it constitutes a post-data preregistration.

**Prior knowledge**: Experiment 1 (binary empathic vs. non-empathic) showed that agents with both emotion observation and empathic reward (alpha=0.5, see_emotions=True) cooperated better than agents with neither (alpha=0.0, see_emotions=False). However, this design confounded two mechanisms: (1) informational access to others' emotional states and (2) reward-based motivation to improve others' well-being. Experiment 3 (multiple alpha) studied the dose-response of the reward component. This experiment isolates each mechanism independently using a factorial design.

---

## 2. Hypotheses

### Primary hypotheses

**H1 — Reward effect (B > A)**: Agents with empathic reward but no emotion observation (condition B: alpha=0.5, see_emotions=False) achieve better resource preservation than baseline agents (condition A: alpha=0.0, see_emotions=False).

Operationalization: One-sided comparison on each primary DV — B shows lower depletion and longer episodes than A.

**H2 — Observation effect (C > A)**: Agents that observe emotions but receive no empathic reward (condition C: alpha=0.0, see_emotions=True) achieve better resource preservation than baseline (condition A).

Operationalization: One-sided comparison on each primary DV — C shows lower depletion and longer episodes than A.

**H3 — Full empathy superiority (D > B and D > C)**: Agents with both observation and empathic reward (condition D: alpha=0.5, see_emotions=True) outperform agents with only one component (conditions B and C).

Operationalization: Two one-sided comparisons — D > B and D > C on each primary DV.

**H4 — Super-additive interaction (synergy)**: The combined effect of observation and empathic reward is greater than the sum of their individual effects. Formally, the interaction term in a 2x2 ANOVA is significant and positive (super-additive).

Operationalization: Significant interaction effect in a 2x2 ANOVA (see_emotions x alpha) for each primary DV. The improvement D-A exceeds (B-A) + (C-A).

### Exploratory hypotheses

**H5 — Inequality reduction**: Full empathy (condition D) reduces reward inequality (Gini coefficient) relative to all other conditions.

Operationalization: Pairwise comparisons of Gini — D vs. {A, B, C}.

---

## 3. Variables

### Independent variables (2x2 factorial)

1. **see_emotions** (observation dimension): {True, False}
   - True: agents observe the mean emotion of the other 5 agents (scalar input)
   - False: agents receive a zero vector (no informational access)

2. **alpha** (reward dimension): {0.0, 0.5}
   - 0.0: reward is purely personal satisfaction
   - 0.5: reward is 50% personal satisfaction + 50% empathic reward (mean emotion of others)

### Four experimental conditions

| Condition | Label | see_emotions | alpha | Description |
| --- | --- | --- | --- | --- |
| A | blind_non_empathic | False | 0.0 | Baseline: no observation, no empathic reward |
| B | blind_reward_empathic | False | 0.5 | Empathic reward without observational access |
| C | sees_ignores | True | 0.0 | Observes emotions but reward is purely personal |
| D | full_empathy | True | 0.5 | Full empathy: observation + empathic reward |

### Primary dependent variables

1. **Resource depletion (final)**: `1 - (resource_remaining / 500)` at episode end. Range: [0, 1]. Higher = worse preservation.

2. **Resource depletion (cumulative)**: `1 - (mean_resource_across_steps / 500)`. Captures resource trajectory over the full episode.

3. **Episode length**: `total_steps` per episode. Range: [1, 1000]. Longer = better preservation.

4. **Gini coefficient**: Computed from the 6 agents' total personal rewards per episode. Range: [0, 1]. Higher = more unequal.

### Exploratory dependent variables

5. **Cooperation rate**: Fraction of abstain actions (action=0) across all agents and steps per episode.

6. **Per-agent efficiency**: Mean personal reward per agent per episode.

### Fixed parameters (not varied)

| Parameter | Value | Description |
| --- | --- | --- |
| beta | 0.5 | Weight of last meal vs. history in personal satisfaction |
| smoothing | linear | Emotion mapping function |
| threshold | 0.5 | Neutral emotion threshold |
| emotion_decimals | 2 | Rounding precision for emotion signal |
| emotion_type | average | Agents observe mean emotion (scalar) |

---

## 4. Experimental Design and Paradigm

### Study type

Computational simulation experiment. 2x2 between-condition factorial design.

### Environment: Tragedy of the Commons

- **Agents**: 6 DQN agents sharing a common resource pool
- **Initial resources**: 500 units
- **Resource dynamics**: Stochastic exploitation — success probability = `resource / 500`. Each successful exploitation removes 1 unit.
- **Regeneration**: `new_resource = max(0, (resource - consumed) * regen_rate)` with regen_rate = 1.0
- **Actions**: Binary — exploit (1) or abstain (0)
- **Episode termination**: Resource reaches 0 or max steps (1000) reached
- **Episodes per run**: 500
- **Homogeneous conditions**: All 6 agents within a run share the same see_emotions and alpha values

### Agent architecture: Deep Q-Network (DQN)

- Input: average emotion of other agents (1-dimensional) when see_emotions=True; zero vector when see_emotions=False
- Hidden layers: 2 layers, 64 neurons each, ReLU activation
- Output: Q-values for 2 actions (abstain, exploit)
- Experience replay buffer
- Target network updated every 5 steps
- Hyperparameters: learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, batch_size=16

### Per-agent attribute implementation

Unlike Experiments 1-3 where see_emotions and alpha were global environment flags, this experiment implements them as **per-agent attributes** (`agent.see_emotions`, `agent.alpha`). This enables the factorial design and supports future mixed-population extensions. Within this experiment, all agents in a given run share the same configuration.

### Emotion and reward calculation

**Emotion**: Each agent tracks a sliding window of 10 recent meals (binary). Consumption rate = sum(meals) / 10. Emotion is linearly mapped to [-1, 1] based on threshold (0.5):
- rate >= 0.5: emotion = (rate - 0.5) / 0.5
- rate < 0.5: emotion = -(0.5 - rate) / 0.5

**Personal satisfaction**: `beta * last_meal + (1 - beta) * history_rate` (beta=0.5)

**Empathic reward**: Mean emotion of the other 5 agents (computed for all agents regardless of condition, but only weighted in combined reward when alpha > 0)

**Combined reward**: `(1 - alpha) * personal + alpha * empathic`

**Observation gating**: The environment's `get_observations()` method returns:
- `see_emotions=True`: average emotion of other agents (scalar)
- `see_emotions=False`: zero vector of same dimensionality

---

## 5. Sample

- **Conditions**: 4 (2x2 factorial)
- **Runs per condition**: 3 (independent random seeds)
- **Total simulations**: 4 x 3 = 12
- **Episodes per run**: 500
- **Total episode-level observations**: 6,000

**Sample size justification**: 3 runs per condition is consistent with all prior experiments in this project. Each run produces 500 episodes, yielding 1,500 episode-level data points per condition. This sample size is constrained by computational cost.

**Seeding**: Each run uses a deterministic seed via `set_global_seed()`, fixing numpy, random, and torch for reproducibility.

---

## 6. Statistical Analysis Plan

### Unit of analysis

Episode-level DVs aggregated per condition. For ANOVA (N=3 per cell), each run's mean across 500 episodes serves as the observation.

### Tests for H1 (B > A: reward effect)

For each primary DV:
- **One-sided Mann-Whitney U test**: B > A (N=3 per group at run level)
- **One-sided independent t-test**: B > A (supplementary, assuming approximate normality of run means)
- Bonferroni correction across the 4 planned contrasts (H1-H3): alpha_corrected = 0.05 / 4 = 0.0125

### Tests for H2 (C > A: observation effect)

Same approach as H1: one-sided Mann-Whitney and t-test, C > A.

### Tests for H3 (D > B and D > C: full empathy superiority)

Two one-sided comparisons per DV:
- D > B (Mann-Whitney + t-test)
- D > C (Mann-Whitney + t-test)
- Both included in the Bonferroni correction (4 contrasts total)

### Tests for H4 (interaction / synergy)

- **2x2 ANOVA**: see_emotions (True/False) x alpha (0.0/0.5) for each primary DV
  - Main effect of see_emotions
  - Main effect of alpha
  - Interaction term (see_emotions x alpha)
- A significant positive interaction indicates super-additive synergy
- Supplementary: compute interaction contrast = (D - C) - (B - A), test whether > 0

### Tests for H5 (exploratory: inequality reduction)

- Pairwise comparisons of Gini: D vs. A, D vs. B, D vs. C
- No multiple comparison correction (exploratory)

### Learning trajectory analysis

- Windowed moving averages of combined reward per episode per condition
- Mean and min-max envelope across 3 runs per condition
- Visual comparison of learning speed and asymptotic performance across 4 conditions

### Multiple comparison corrections

- **Planned contrasts (H1-H3)**: Bonferroni correction across 4 contrasts per DV (alpha_corrected = 0.05 / 4 = 0.0125)
- **ANOVA (H4)**: Tested separately; alpha = 0.05
- **Across DVs**: Bonferroni correction across the 3 primary DVs for omnibus conclusions (alpha_corrected = 0.05 / 3 = 0.0167)
- **Exploratory analyses (H5)**: No correction; reported as exploratory

### Significance threshold

Alpha = 0.05 (before correction).

### Software

- Python 3.x
- scipy (Mann-Whitney, t-tests)
- statsmodels (ANOVA)
- matplotlib / seaborn (visualization)
- numpy, pandas (data manipulation)

### Missing data

Simulations are deterministic given a seed and run to completion. No missing data is expected. If a simulation file is missing or corrupted, it will be excluded and reported.

### Outlier handling

All 500 episodes per run are included in the analysis. No episodes are excluded.

---

## 7. Exploratory Analyses

The following analyses are explicitly labeled as exploratory and will not be used for confirmatory inference:

1. **Cooperation rate across conditions**: Does abstention frequency differ across the 4 conditions? Is observation alone (C) sufficient to increase restraint?

2. **Condition C learning dynamics**: Do agents in condition C (sees_ignores) learn to use the emotion signal as a proxy for resource state, despite receiving no explicit empathic reward? Analysis of the correlation between observed emotion and chosen action over episodes.

3. **Condition B internal dynamics**: How do agents in condition B (blind_reward_empathic) learn to cooperate without informational access? Is the empathic reward sufficient to shape cooperative behavior via indirect feedback?

4. **Per-agent reward distributions**: Within each condition, how are rewards distributed across the 6 agents? Do some agents specialize as "cooperators" (frequent abstainers) while others free-ride?

5. **Resource depletion trajectories**: Episode-by-episode resource curves per condition, showing how quickly agents learn to preserve resources.

6. **Learning speed comparison**: At which episode does each condition first achieve sustained cooperation (e.g., episode length > 800 for 10 consecutive episodes)?

7. **Comparison with Experiments 1 and 3**: Qualitative comparison of conditions A and D with the original binary experiment (Exp 1), and condition D with alpha=0.5 from Experiment 3.
