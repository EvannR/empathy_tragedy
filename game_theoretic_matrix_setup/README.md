# Game-Theoretic Matrix Setup — 2×2 Empathy Experiment

## Overview

This folder implements a **2×2 factorial experiment** that disentangles two
independent dimensions of empathy in a multi-agent tragedy-of-the-commons
scenario. It extends the `game_theoretic_setup` by independently controlling
whether agents can *observe* others' emotions and whether they *care* about
those emotions in their reward.

---

## Scientific Motivation

The original `game_theoretic_setup` tested two conditions:

| Condition         | `see_emotions` | `alpha` |
|-------------------|---------------|---------|
| Non-empathic      | `False`       | `0.0`   |
| Empathic          | `True`        | `0.5`   |

**Problem:** `see_emotions` and `alpha` were always changed together. This
confounds two mechanistically distinct effects:

- **Observation effect** (`see_emotions`): Can the agent use others' emotional
  states as an informative cue about resource depletion?
- **Reward effect** (`alpha`): Does the agent's objective function incorporate
  others' well-being?

An agent could, in principle, observe emotions without being rewarded for them
(e.g., using them as a predictive signal only), or be rewarded for others'
well-being without being able to observe their current emotional state directly.
Conflating these two effects prevents isolating their contributions to
collective resource conservation.

---

## The 2×2 Design

The two dimensions are crossed to produce four homogeneous conditions:

|                         | `alpha = 0.0` (no empathic reward) | `alpha = 0.5` (empathic reward) |
|-------------------------|------------------------------------|---------------------------------|
| `see_emotions = False`  | **A — Baseline** | **B — Blind Altruist** |
| `see_emotions = True`   | **C — Indifferent Observer** | **D — Full Empathy** |

### Condition descriptions

**A — `blind_non_empathic`** (`see_emotions=False`, `alpha=0.0`)
- Agents receive zero observations and are rewarded purely for their own
  consumption history. Replicates the original non-empathic condition.
  Serves as the baseline.

**B — `blind_reward_empathic`** (`see_emotions=False`, `alpha=0.5`)
- Agents cannot observe others' emotions (zero observation), but their reward
  function still weights others' mean emotion at `alpha=0.5`. Agents are
  intrinsically motivated to improve others' well-being but have no signal to
  condition their policy on. Tests whether reward shaping alone, without
  observational access, produces cooperative behavior.

**C — `sees_ignores`** (`see_emotions=True`, `alpha=0.0`)
- Agents observe the average emotion of other agents (a proxy for resource
  depletion state), but `alpha=0` means their reward is purely personal. Tests
  whether access to the emotional signal of others is informative enough that
  agents learn to use it even without an explicit empathic reward.

**D — `full_empathy`** (`see_emotions=True`, `alpha=0.5`)
- Agents both observe others' emotions and weight them in their reward. Full
  empathic condition. Replicates the original empathic condition.

---

## Hypotheses

- **H1 (Reward effect):** B > A — empathic reward shaping alone improves
  collective outcomes even without observational access to others' emotions.
- **H2 (Observation effect):** C > A — observing others' emotional states is
  itself informative and leads to learned cooperative behavior, even when
  `alpha=0`.
- **H3 (Interaction):** D > B and D > C — observation and reward-shaping have
  complementary (possibly synergistic) effects; neither alone is sufficient to
  reach the cooperative level achieved by their combination.

---

## Key Differences from `game_theoretic_setup`

### 1. Per-agent `see_emotions` and `alpha`
In `game_theoretic_setup`, `see_emotions` was an environment-level flag shared
by all agents, and `alpha` was a single scalar in `SocialRewardCalculator`.

In this setup:
- `see_emotions` and `alpha` are **per-agent attributes** stored on each agent
  instance (`agent.see_emotions`, `agent.alpha`).
- `SocialRewardCalculator` accepts `alpha` as a list of per-agent values and
  computes: `r_i = (1 - alpha_i) * personal_i + alpha_i * empathic_i`.
- The environment reads per-agent `see_emotions` flags to decide, agent by
  agent, whether to return real emotion observations or a zero vector.

This design supports future **mixed-population experiments** where some agents
in the same simulation are empathic and others are not.

### 2. Decoupled observation and reward
The four conditions isolate what was previously a single binary:
- Emotion signal in observation space → independent of reward weighting.
- Empathic term in reward function → independent of observation.

### 3. Four conditions instead of two
The main script iterates over `EMPATHY_CONDITIONS`, a list of
`(see_emotions, alpha, label)` triples covering all four cells of the matrix.

### 4. Output directory
Results are written to `../GT_simulation_matrix/` (sibling of this folder).
Filenames encode both dimensions: `..._<see_emotions>_<alpha>_...` so the
four conditions can be distinguished when loading CSVs for analysis.

---

## File Structure

```
game_theoretic_matrix_setup/
  agent_policies_game_theoretic.py   # Agent base class now carries see_emotions and alpha;
                                     # SocialRewardCalculator accepts per-agent alpha list.
  env_game_theoretic.py              # get_observations() gates per agent using agent.see_emotions;
                                     # _init_agents() reads per-agent configs.
  main_game_theoretic_new_version.py # 4-condition matrix experiment runner.
  value_cal.py                       # Small reward formula utility (unchanged).
  README.md                          # This file.
```

---

## Running the Experiment

```bash
cd game_theoretic_matrix_setup
python main_game_theoretic_new_version.py
```

Results are saved incrementally to `../GT_simulation_matrix/`.
Each condition produces `NUM_RUNS_PER_CONDITION` episode-summary CSV files.

### Quick sanity check

Set `EPISODE_NUMBER = 1` and `NUM_RUNS_PER_CONDITION = 1` to verify all four
conditions run without errors and produce correctly formatted CSV files before
launching a full experiment.
