# Reward Design in Pig Dice

Reinforcement learning case study comparing reward design techniques in the dice game **Pig**. Agents are trained via **MC Control** and **Q-Learning** against the optimal policy (computed by Value Iteration), and their convergence is measured under different reward signals.

## Reward Methods

| Method | Description |
|--------|-------------|
| **Sparse** | Latent +1/−1 win/loss signal (baseline) |
| **PBRS (Φ₁)** | Potential-based shaping on banked-score lead |
| **PBRS (Φ₂)** | Potential-based shaping on turn total |
| **Naive Curiosity** | Prediction-error intrinsic motivation |
| **ICM** | Intrinsic Curiosity Module (latent-space filtered) |
| **Count-Based** | Visitation-count exploration bonus |
| **IRL** | MaxEnt Inverse RL from expert trajectories |

## Quick Start

```bash
pip install -r requirements.txt

# 1. Compute and cache the optimal policy via Value Iteration
python train.py --run-vi

# 2. Verify PBRS policy invariance (runs VI on shaped rewards)
python train.py --verify-pbrs

# 3. Run IRL to recover reward weights (saves irl_weights.json)
python train.py --run-irl

# 4. Run all reward experiments (4 seeds each, 1M episodes)
python train.py

# 5. Run a single experiment
python train.py --reward sparse --agent ql --episodes 500000
```

## Project Structure

```
├── train.py                          Main entry point
├── pig_dice_env.py                   Gymnasium-style Pig Dice environment
├── agents.py                         MC Control and Q-Learning agents
├── rewards.py                        Reward function implementations
├── value_iteration.py                Optimal policy via Value Iteration
├── irl.py                            Maximum Entropy IRL
├── evaluate.py                       Win rate evaluation and policy deviation
├── config.py                         Hyperparameters and experiment configuration
├── plot_results.py                   Standalone plotting from saved CSVs
├── st_petersburg.py                  St. Petersburg Paradox utility experiments
├── reward_density_and_policy_deviation.py  PBRS policy invariance verification
├── requirements.txt
└── README.md
```

## Command Reference

```bash
# Value Iteration
python train.py --run-vi

# Verify PBRS policy invariance via Value Iteration
python train.py --verify-pbrs

# IRL weight recovery
python train.py --run-irl

# Reward density sampling
python train.py --density

# Hyperparameter search (coarse-to-fine)
python train.py --hypersearch-zoom --hypersearch-agent mc
python train.py --hypersearch-zoom --hypersearch-agent ql

# Train with best hyperparameters from search
python train.py --train-best --hypersearch-csv hypersearch_results.csv

# Regenerate plots from saved CSVs
python plot_results.py --results-dir precomputed_results/pig_dice/

# St. Petersburg Paradox
python train.py --run-st-pete
```

## Hyperparameters

Fixed across all reward experiments (selected via coarse-to-fine grid search on the sparse baseline):

| Parameter | MC Control | Q-Learning |
|-----------|-----------|------------|
| Optimistic Init *k* | 0.5 | 0.75 |
| Initial ε₀ | 0.5625 | 0.5 |
| Decay rate μ | 0.9999999 | 0.9999999 |
| Learning rate α₀ | — | 0.5 |
| Decay control *c* | — | 500,000 |
| Training episodes | 1,000,000 | 1,000,000 |

## Key Results

| Reward | MC Control (%) | Q-Learning (%) |
|--------|---------------|----------------|
| Sparse (baseline) | 33.2 | 40.7 |
| PBRS-Φ₁ | — | **41.5+** |
| PBRS-Φ₂ | — | — |
| Naive Curiosity | — | — |
| ICM | **~30+** | — |
| Count-Based | **~33+** | — |
| IRL (Optimal, N=200) | **~35** | **~39** |
| IRL (Optimal, N=2000) | — | — |

See `precomputed_results/pig_dice/` for Pig Dice learning curves and `precomputed_results/st_petersburg/` for utility plots.
