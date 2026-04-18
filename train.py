"""
Main training script for Pig Dice reward design experiments.

Usage:
    python train.py                                    # run all reward experiments
    python train.py --reward sparse --agent ql
    python train.py --run-vi                           # compute optimal policy
    python train.py --verify-pbrs                      # verify PBRS policy invariance
    python train.py --run-irl                          # recover IRL weights
    python train.py --run-st-pete                      # St. Petersburg Paradox
    python train.py --run-hypersearch --hypersearch-agent mc
    python train.py --hypersearch-zoom --hypersearch-agent mc
    python train.py --train-best --hypersearch-csv hypersearch_results.csv
    python train.py --plot --curves-csv learning_curves.csv
    python train.py --density                          # sample reward density
"""

import argparse
import csv
import os
import sys
import time
import numpy as np
from collections import deque

from pig_dice_env import PigDiceEnv, ROLL, HOLD, TARGET
from agents import MCControlAgent, QLearningAgent
from rewards import make_reward
from evaluate import evaluate_win_rate
from value_iteration import (
    value_iteration, make_optimal_policy_fn,
    value_iteration_shaped, load_policy, save_policy,
)
import config


# ======================================================================
# Optimal Policy
# ======================================================================
def compute_or_load_optimal_policy(force_recompute=False, verbose=True):
    cache_path = "optimal_policy.npz"
    if not force_recompute and os.path.exists(cache_path):
        if verbose:
            print("Loading cached optimal policy...")
        pi, P = load_policy(cache_path)
        if verbose:
            print(f"  P(0,0,0) = {P[0,0,0]:.4f}")
            print(f"  Policy size: {len(pi)} states")
        return P, pi

    if verbose:
        print("Computing optimal policy via Value Iteration...")
        print("  (This takes ~2-3 minutes. Result is cached for future runs.)")
    t0 = time.time()
    _, pi, P = value_iteration(verbose=verbose, tol=0.005)
    elapsed = time.time() - t0
    if verbose:
        print(f"Value Iteration completed in {elapsed:.1f}s")
        print(f"P(0,0,0) = {P[0, 0, 0]:.4f}")

    save_policy(pi, P, cache_path)
    return P, pi


# ======================================================================
# PBRS Policy Invariance Verification
# ======================================================================
def verify_pbrs_policy_invariance(optimal_pi, verbose=True):
    """Run VI on each shaped reward and report d(pi*_R', pi*_R)."""
    from evaluate import compute_policy_deviation
    from rewards import PBRSReward, PBRS2Reward

    print("\n" + "=" * 60)
    print("PBRS POLICY INVARIANCE VERIFICATION")
    print("=" * 60)

    for label, reward_cls in [("Phi_1", PBRSReward), ("Phi_2", PBRS2Reward)]:
        reward_fn = reward_cls(gamma=1.0)
        print(f"\n  Running Value Iteration on R' = R + F ({label})...")
        t0 = time.time()
        pi_shaped = value_iteration_shaped(reward_fn, verbose=verbose)
        elapsed = time.time() - t0
        print(f"  VI completed in {elapsed:.1f}s")

        deviation, n_disagree, n_total = compute_policy_deviation(
            pi_shaped, optimal_pi
        )
        print(f"  d(pi*_R', pi*_R) = {deviation:.6f}  "
              f"({n_disagree}/{n_total} states disagree)")
        if deviation == 0.0:
            print(f"  -> Policy invariance VERIFIED for {label}.")
        else:
            print(f"  -> WARNING: policy deviation is non-zero for {label}!")


# ======================================================================
# Core Training Loop
# ======================================================================
def train_agent(agent_type, reward_fn, opponent_policy_fn, n_episodes,
                log_interval=1000, eval_interval=None, eval_episodes=None,
                agent_config=None, run_name="", verbose=True, target_wr=None):
    """Train one agent; return (agent, metrics, rolling_history, greedy_history)."""
    if agent_type == "mc":
        cfg = {**config.MC_CONFIG, **(agent_config or {})}
        agent = MCControlAgent(**cfg)
    elif agent_type == "ql":
        cfg = {**config.QL_CONFIG, **(agent_config or {})}
        agent = QLearningAgent(**cfg)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    env = PigDiceEnv(opponent_policy=opponent_policy_fn, agent_starts=True)

    _eval_interval = eval_interval or config.GREEDY_EVAL_INTERVAL
    _eval_episodes = eval_episodes or config.GREEDY_EVAL_EPISODES

    win_history = deque(maxlen=_eval_interval)
    rolling_history = []   # (episode, win_rate)
    greedy_history = []    # (episode, greedy_win_rate)
    density_sum = 0.0
    density_count = 0
    prefix = run_name

    for episode in range(1, n_episodes + 1):
        obs, _ = env.reset()
        done = False
        trajectory = []
        ep_nonzero = 0
        ep_steps = 0

        while not done:
            state = tuple(obs)
            action = agent.select_action(state)
            obs_next, env_reward, done, _, _ = env.step(action)
            state_next = tuple(obs_next)
            shaped_reward = reward_fn(state, action, state_next, env_reward)

            if agent_type == "mc":
                trajectory.append((state, action, shaped_reward))
            elif agent_type == "ql":
                agent.update_step(state, action, shaped_reward, state_next, done)

            agent.decay_epsilon()
            if shaped_reward != 0 or env_reward != 0:
                ep_nonzero += 1
            ep_steps += 1
            obs = obs_next

        if agent_type == "mc" and trajectory:
            agent.update_episode(trajectory)

        won = 1 if env.winner == "agent" else 0
        win_history.append(won)

        if ep_steps > 0:
            density_sum += ep_nonzero / ep_steps
            density_count += 1

        if episode % log_interval == 0:
            recent_wr = sum(win_history) / max(len(win_history), 1)
            rolling_history.append((episode, recent_wr))
            if verbose and episode % (log_interval * 10) == 0:
                print(f"  [{prefix}] Episode {episode:>8d} | "
                      f"WR: {recent_wr:.3f} | eps: {agent.epsilon:.6f}")

        if episode % _eval_interval == 0:
            greedy_wr, _ = evaluate_win_rate(
                agent, opponent_policy_fn, n_episodes=_eval_episodes, seed=episode
            )
            greedy_history.append((episode, greedy_wr))
            print(f"  [{prefix}] ep {episode:>9,} | greedy WR: {greedy_wr:.4f}",
                  flush=True)
            if target_wr is not None and greedy_wr >= target_wr:
                print(f"  [{prefix}] Target WR {target_wr:.4f} reached.", flush=True)
                break

    if verbose:
        print(f"  [{prefix}] Training complete. Running final evaluation...")

    final_wr, eval_info = evaluate_win_rate(
        agent, opponent_policy_fn, n_episodes=config.EVAL_EPISODES
    )

    metrics = {
        "final_win_rate": final_wr,
        "avg_steps_per_game": eval_info["avg_steps"],
        "mean_reward_density": density_sum / density_count if density_count > 0 else 0.0,
    }

    if verbose:
        print(f"  [{prefix}] Final WR: {final_wr:.4f}")

    return agent, metrics, rolling_history, greedy_history


# ======================================================================
# Multi-seed Training
# ======================================================================
def _seed_wrapper(args):
    """Picklable worker: train one seed."""
    (seed, agent_type, reward_name, reward_kwargs, agent_config,
     optimal_pi, n_episodes, log_interval, eval_interval, eval_episodes,
     target_wr) = args

    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)

    from rewards import make_reward
    from value_iteration import make_optimal_policy_fn

    reward_fn = make_reward(reward_name, **reward_kwargs)
    opponent_fn = make_optimal_policy_fn(optimal_pi)

    _, metrics, rolling_history, greedy_history = train_agent(
        agent_type=agent_type,
        reward_fn=reward_fn,
        opponent_policy_fn=opponent_fn,
        n_episodes=n_episodes,
        log_interval=log_interval,
        eval_interval=eval_interval,
        eval_episodes=eval_episodes,
        agent_config=agent_config,
        run_name=f"seed{seed}",
        verbose=False,
        target_wr=target_wr,
    )
    return seed, rolling_history, greedy_history, metrics


def train_agent_multi_seed(agent_type, reward_name, reward_kwargs, optimal_pi,
                           n_episodes, n_seeds=None, n_workers=None,
                           log_interval=1000, eval_interval=None,
                           eval_episodes=None, agent_config=None,
                           run_name="", target_wr=None):
    """Train n_seeds agents in parallel; return aggregated metrics dict."""
    from concurrent.futures import ProcessPoolExecutor

    _n_seeds = n_seeds or config.N_SEEDS
    _eval_interval = eval_interval or config.GREEDY_EVAL_INTERVAL
    _eval_episodes = eval_episodes or config.GREEDY_EVAL_EPISODES

    seeds = [42 + i for i in range(_n_seeds)]
    worker_args = [
        (seed, agent_type, reward_name, reward_kwargs, agent_config,
         optimal_pi, n_episodes, log_interval, _eval_interval, _eval_episodes,
         target_wr)
        for seed in seeds
    ]

    _max_workers = min(n_workers, _n_seeds) if n_workers is not None else _n_seeds

    if _max_workers <= 2:
        print(f"  [{run_name}] Running {_n_seeds} seeds sequentially...")
        results = [_seed_wrapper(a) for a in worker_args]
    else:
        print(f"  [{run_name}] Launching {_n_seeds} seeds ({_max_workers} workers)...")
        with ProcessPoolExecutor(max_workers=_max_workers) as executor:
            results = list(executor.map(_seed_wrapper, worker_args))

    all_greedy = [r[2] for r in results]
    all_metrics = [r[3] for r in results]

    min_len = min(len(g) for g in all_greedy)
    greedy_agg = []
    for i in range(min_len):
        ep = all_greedy[0][i][0]
        wrs = [all_greedy[s][i][1] for s in range(_n_seeds)]
        greedy_agg.append({
            "episode": ep,
            "mean_wr": float(np.mean(wrs)),
            "std_wr": float(np.std(wrs)),
        })

    final_wrs = [m["final_win_rate"] for m in all_metrics]
    densities = [m["mean_reward_density"] for m in all_metrics]

    agg_metrics = {
        "final_win_rate": float(np.mean(final_wrs)),
        "final_win_rate_std": float(np.std(final_wrs)),
        "mean_reward_density": float(np.mean(densities)),
        "greedy_history": greedy_agg,
        "n_seeds": _n_seeds,
    }

    print(f"  [{run_name}] Final WR: {agg_metrics['final_win_rate']:.4f}"
          f" +/- {agg_metrics['final_win_rate_std']:.4f} | "
          f"Density: {agg_metrics['mean_reward_density']:.5f}")
    return agg_metrics


# ======================================================================
# IRL Experiments
# ======================================================================
def run_irl_experiments(optimal_pi, opponent_policy_fn,
                        weights_file="irl_weights.json"):
    from irl import maxent_irl, generate_expert_trajectories
    import json

    print("\n" + "=" * 60)
    print("INVERSE REINFORCEMENT LEARNING EXPERIMENTS")
    print("=" * 60)

    def opt_fn(s_A, s_B, kappa):
        return optimal_pi.get((s_A, s_B, kappa), ROLL)

    expert_configs = {
        "optimal":     {"policy_fn": opt_fn, "epsilon": 0.0},
        "sub_optimal": {"policy_fn": opt_fn, "epsilon": 0.05},
    }

    saved_weights = {}

    for expert_name, expert_cfg in expert_configs.items():
        for n_traj in config.IRL_CONFIG["n_expert_trajectories"]:
            run_key = f"irl_{expert_name}_N{n_traj}"
            print(f"\n--- {run_key} ---")

            trajectories = generate_expert_trajectories(
                policy_fn=expert_cfg["policy_fn"],
                n_trajectories=n_traj,
                opponent_policy=opponent_policy_fn,
                epsilon=expert_cfg["epsilon"],
            )
            print(f"  Generated {n_traj} trajectories ({expert_name})")

            theta, history = maxent_irl(
                trajectories,
                alpha=config.IRL_CONFIG["alpha"],
                alpha_decay=config.IRL_CONFIG["alpha_decay"],
                max_outer_iters=config.IRL_CONFIG["max_outer_iters"],
                tol=config.IRL_CONFIG["tol"],
                n_eval_episodes=config.IRL_CONFIG["n_eval_episodes"],
                verbose=True,
            )
            print(f"  Recovered theta = {theta}")
            saved_weights[run_key] = theta.tolist()

    with open(weights_file, "w") as f:
        json.dump(saved_weights, f, indent=2)
    print(f"\nIRL weights saved to: {weights_file}")
    print("Keys:", list(saved_weights.keys()))
    return saved_weights


# ======================================================================
# Best-Hyperparameter Training
# ======================================================================
def _train_to_target_worker(args):
    """Picklable worker: train one seed to target WR or max episodes."""
    (seed, agent_type, agent_cfg, optimal_pi,
     max_episodes, eval_interval, eval_episodes, target_wr) = args

    import random, numpy as np
    from rewards import make_reward
    from value_iteration import make_optimal_policy_fn
    from agents import MCControlAgent, QLearningAgent
    from pig_dice_env import PigDiceEnv
    from evaluate import evaluate_win_rate

    random.seed(seed)
    np.random.seed(seed)
    sparse_fn = make_reward("sparse")
    opponent_fn = make_optimal_policy_fn(optimal_pi)

    agent = (MCControlAgent(**agent_cfg) if agent_type == "mc"
             else QLearningAgent(**agent_cfg))
    env = PigDiceEnv(opponent_policy=opponent_fn, agent_starts=True)

    curve = []
    total_episodes = 0

    while total_episodes < max_episodes:
        for _ in range(eval_interval):
            obs, _ = env.reset()
            done = False
            trajectory = []
            while not done:
                state = tuple(obs)
                action = agent.select_action(state)
                obs_next, env_reward, done, _, _ = env.step(action)
                state_next = tuple(obs_next)
                shaped = sparse_fn(state, action, state_next, env_reward)
                if agent_type == "mc":
                    trajectory.append((state, action, shaped))
                else:
                    agent.update_step(state, action, shaped, state_next, done)
                agent.decay_epsilon()
                obs = obs_next
            if agent_type == "mc" and trajectory:
                agent.update_episode(trajectory)

        total_episodes += eval_interval
        wr, _ = evaluate_win_rate(
            agent, opponent_fn, n_episodes=eval_episodes,
            seed=seed + total_episodes,
        )
        curve.append((total_episodes, wr))
        if wr >= target_wr:
            break

    return seed, curve


def train_to_target(agent_type, agent_cfg, optimal_pi, n_seeds=None,
                    max_episodes=None, eval_interval=None, eval_episodes=None,
                    target_wr=None):
    """Train n_seeds agents until each reaches target_wr or max_episodes; return aggregated curves."""
    from concurrent.futures import ProcessPoolExecutor

    _n_seeds = n_seeds or config.TRAIN_N_SEEDS
    _max_episodes = max_episodes or config.TRAIN_MAX_EPISODES
    _eval_interval = eval_interval or config.TRAIN_GREEDY_INTERVAL
    _eval_episodes = eval_episodes or config.TRAIN_GREEDY_EVAL_EPS
    _target_wr = target_wr if target_wr is not None else config.TARGET_WIN_RATE

    seeds = [42 + i for i in range(_n_seeds)]
    worker_args = [
        (seed, agent_type, agent_cfg, optimal_pi,
         _max_episodes, _eval_interval, _eval_episodes, _target_wr)
        for seed in seeds
    ]

    print(f"  [{agent_type.upper()}] Launching {_n_seeds} seeds "
          f"(max {_max_episodes:,} eps, target WR={_target_wr})...")

    with ProcessPoolExecutor(max_workers=_n_seeds) as executor:
        results = list(executor.map(_train_to_target_worker, worker_args))

    seed_curves = {seed: dict(curve) for seed, curve in results}
    common_eps = sorted(set.intersection(*[set(d) for d in seed_curves.values()]))

    curves = []
    for ep in common_eps:
        seed_wrs = [seed_curves[s][ep] for s in sorted(seed_curves)]
        curves.append({
            "episode": ep,
            "mean_wr": float(np.mean(seed_wrs)),
            "std_wr": float(np.std(seed_wrs)),
        })
        print(f"  [{agent_type.upper()}] Episode {ep:>10,} | "
              f"Greedy WR: {curves[-1]['mean_wr']:.4f} +/- {curves[-1]['std_wr']:.4f}")

    return curves


def run_best_training(hypersearch_csv, optimal_pi, opponent_fn, n_seeds=None,
                      max_episodes=None, eval_interval=None, eval_episodes=None,
                      target_wr=None, output_csv="learning_curves.csv",
                      output_pdf="learning_curves.pdf"):
    """Load best MC/QL configs from hypersearch CSV, train to target WR, save curves."""
    mc_best = ql_best = None
    with open(hypersearch_csv, newline="") as f:
        for row in csv.DictReader(f):
            wr = float(row["final_win_rate"])
            agent = row["agent"]
            if agent == "mc" and (mc_best is None or wr > float(mc_best["final_win_rate"])):
                mc_best = row
            elif agent == "ql" and (ql_best is None or wr > float(ql_best["final_win_rate"])):
                ql_best = row

    if mc_best is None or ql_best is None:
        raise ValueError(f"Missing MC or QL rows in {hypersearch_csv}")

    mc_keys = ["epsilon_0", "decay_rate", "optimistic_init"]
    ql_keys = ["epsilon_0", "decay_rate", "optimistic_init", "alpha_0", "decay_control"]

    mc_cfg = {**config.MC_CONFIG}
    for k in mc_keys:
        if k in mc_best:
            mc_cfg[k] = float(mc_best[k])

    ql_cfg = {**config.QL_CONFIG}
    for k in ql_keys:
        if k in ql_best:
            ql_cfg[k] = float(ql_best[k])

    print(f"\nBest MC (WR={float(mc_best['final_win_rate']):.4f}):")
    for k in mc_keys:
        print(f"  {k} = {mc_cfg[k]}")
    print(f"\nBest QL (WR={float(ql_best['final_win_rate']):.4f}):")
    for k in ql_keys:
        print(f"  {k} = {ql_cfg[k]}")

    kw = dict(n_seeds=n_seeds, max_episodes=max_episodes,
              eval_interval=eval_interval, eval_episodes=eval_episodes,
              target_wr=target_wr)

    print(f"\n{'=' * 60}\nTRAINING MC CONTROL\n{'=' * 60}")
    mc_curves = train_to_target("mc", mc_cfg, optimal_pi, **kw)

    print(f"\n{'=' * 60}\nTRAINING Q-LEARNING\n{'=' * 60}")
    ql_curves = train_to_target("ql", ql_cfg, optimal_pi, **kw)

    all_rows = ([{"agent": "mc", **r} for r in mc_curves] +
                [{"agent": "ql", **r} for r in ql_curves])
    if all_rows:
        cols = list(all_rows[0].keys())
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nLearning curves saved to: {output_csv}")

    _target = target_wr if target_wr is not None else config.TARGET_WIN_RATE
    plot_learning_curves(mc_curves, ql_curves, output_pdf, target_wr=_target)

    return mc_curves, ql_curves


# ======================================================================
# Reward Density Sampling
# ======================================================================
def sample_reward_density(n_episodes=5_000, seed=42):
    """Roll out a random policy for each reward method and print density statistics."""
    np.random.seed(seed)
    _, optimal_pi = compute_or_load_optimal_policy(verbose=False)
    opponent_fn = make_optimal_policy_fn(optimal_pi)

    print(f"\n{'Reward':<20} {'Density (mean +/- std)':>22}  "
          f"{'Non-zero / total steps'}")
    print("-" * 70)

    for cfg in config.REWARD_METHODS:
        reward_name = cfg["name"]
        reward_kwargs = {k: v for k, v in cfg.items() if k != "name"}
        reward_fn = make_reward(reward_name, **reward_kwargs)

        env = PigDiceEnv(opponent_policy=opponent_fn, agent_starts=True)
        ep_densities = []
        total_steps = total_nonzero = 0

        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            ep_steps = ep_nonzero = 0
            while not done:
                action = np.random.randint(2)
                obs_next, env_reward, done, _, _ = env.step(action)
                shaped = reward_fn(tuple(obs), action, tuple(obs_next), env_reward)
                if shaped != 0 or env_reward != 0:
                    ep_nonzero += 1
                ep_steps += 1
                obs = obs_next
            if ep_steps > 0:
                ep_densities.append(ep_nonzero / ep_steps)
            total_steps += ep_steps
            total_nonzero += ep_nonzero

        mean_d = np.mean(ep_densities)
        std_d = np.std(ep_densities)
        label = reward_kwargs.get("theta_key", reward_name)
        print(f"{label:<20} {mean_d:>8.4f} +/- {std_d:<8.4f}   "
              f"{total_nonzero:>8,} / {total_steps:,}")

    print()


# ======================================================================
# CSV Save / Load
# ======================================================================
def save_results_csv(all_results, output_csv):
    """Save greedy checkpoint history for all runs to CSV."""
    rows = []
    for run_name, m in all_results.items():
        density = m.get("mean_reward_density", float("nan"))
        for pt in m.get("greedy_history", []):
            rows.append({
                "run_name": run_name,
                "episode": pt["episode"],
                "mean_wr": pt["mean_wr"],
                "std_wr": pt["std_wr"],
                "mean_reward_density": density,
            })
    if rows:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Results saved to: {output_csv}")


def load_results_csv(path):
    """Load a results CSV into all_results greedy_history format."""
    results = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            run_name = row["run_name"]
            if run_name not in results:
                results[run_name] = {"greedy_history": []}
            results[run_name]["greedy_history"].append({
                "episode": int(row["episode"]),
                "mean_wr": float(row["mean_wr"]),
                "std_wr": float(row["std_wr"]),
            })
    return results


# ======================================================================
# Plotting
# ======================================================================
def plot_learning_curves(mc_curves, ql_curves, output_pdf="learning_curves.pdf",
                         target_wr=None):
    """Save MC (red) and QL (blue) mean greedy win-rate curves to PDF."""
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _target = target_wr if target_wr is not None else config.TARGET_WIN_RATE

    fig, ax = plt.subplots(figsize=(8, 5))

    def _draw(curves, color, label):
        eps = [r["episode"] for r in curves]
        mean = np.array([r["mean_wr"] for r in curves])
        std = np.array([r["std_wr"] for r in curves])
        ax.plot(eps, mean, color=color, linewidth=2, label=label)
        ax.fill_between(eps, mean - std, mean + std, alpha=0.20, color=color)

    if mc_curves:
        _draw(mc_curves, "red", "MC Control")
    if ql_curves:
        _draw(ql_curves, "blue", "Q-Learning")

    ax.axhline(y=_target, color="black", linestyle="--", linewidth=1.2,
               label=f"Target WR = {_target}")
    ax.set_xlabel("Training Episode", fontsize=13)
    ax.set_ylabel("Greedy Win Rate", fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(1.0, _target + 0.1))

    max_ep = max((r["episode"] for r in (mc_curves + ql_curves)), default=0)
    if max_ep >= 500_000:
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M")
        )
        ax.set_xlabel("Training Episode (millions)", fontsize=13)

    plt.tight_layout()
    plt.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {output_pdf}")


def plot_reward_design_curves(all_results, output_pdf="reward_design_curves.pdf",
                              target_wr=None, baseline=None):
    """Save win-rate learning curves for all reward experiments to PDF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    mc_runs = [n for n in all_results if n.endswith("_mc")]
    ql_runs = [n for n in all_results if n.endswith("_ql")]

    def _shade_palette(names, cmap_name, lo=0.40, hi=0.85):
        cmap = plt.cm.get_cmap(cmap_name)
        n = len(names)
        if n == 0:
            return {}
        if n == 1:
            return {names[0]: cmap(0.65)}
        return {name: cmap(lo + (hi - lo) * i / (n - 1))
                for i, name in enumerate(names)}

    color_map = {
        **_shade_palette(mc_runs, "Reds"),
        **_shade_palette(ql_runs, "Blues"),
    }

    def _run_color(run_name):
        return color_map.get(run_name, "red" if run_name.endswith("_mc") else "blue")

    all_sources = list(all_results.values()) + list((baseline or {}).values())
    max_ep = max(
        (pt["episode"] for src in all_sources for pt in src.get("greedy_history", [])),
        default=0,
    )
    use_millions = max_ep >= 500_000

    fig, ax = plt.subplots(figsize=(8, 5))

    def _draw(run_name, history, dashed=False):
        if not history:
            return
        color = _run_color(run_name)
        alpha = 0.45 if dashed else 1.0
        lw = 1.2 if dashed else 1.8
        ls = "--" if dashed else "-"
        label = f"{run_name} (baseline)" if dashed else run_name

        eps = [pt["episode"] for pt in history]
        mean = np.array([pt["mean_wr"] for pt in history])
        std = np.array([pt["std_wr"] for pt in history])

        ax.plot(eps, mean, color=color, linewidth=lw, linestyle=ls,
                alpha=alpha, label=label)
        ax.fill_between(eps, mean - std, mean + std,
                        alpha=0.08 if dashed else 0.15, color=color)

    if baseline:
        for run_name, m in baseline.items():
            _draw(run_name, m.get("greedy_history", []), dashed=True)
    for run_name, m in all_results.items():
        _draw(run_name, m.get("greedy_history", []))

    if target_wr is not None:
        ax.axhline(y=target_wr, color="black", linestyle="--", linewidth=1.0,
                   label=f"Target WR = {target_wr:.4f}")

    ax.set_ylabel("Greedy Win Rate", fontsize=12)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    if use_millions:
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M")
        )
    ax.set_xlabel("Training Episode (millions)" if use_millions else "Training Episode",
                  fontsize=12)

    plt.tight_layout()
    fig.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Win rate curves saved to: {output_pdf}")


def plot_from_csv(curves_csv, output_pdf="learning_curves.pdf", target_wr=None):
    """Regenerate learning-curve PDF from a saved CSV."""
    mc_curves, ql_curves = [], []
    with open(curves_csv, newline="") as f:
        for row in csv.DictReader(f):
            d = {"episode": int(row["episode"]),
                 "mean_wr": float(row["mean_wr"]),
                 "std_wr": float(row["std_wr"])}
            (mc_curves if row["agent"] == "mc" else ql_curves).append(d)
    plot_learning_curves(mc_curves, ql_curves, output_pdf, target_wr=target_wr)


# ======================================================================
# Hyperparameter Search
# ======================================================================
def _hs_worker(args):
    """Picklable worker: evaluate one hyperparameter combination."""
    agent_type, keys, values, optimal_pi, n_episodes = args

    from rewards import make_reward
    from value_iteration import make_optimal_policy_fn

    hparams = dict(zip(keys, values))
    sparse_fn = make_reward("sparse")
    opponent_fn = make_optimal_policy_fn(optimal_pi)

    _, metrics, _, _ = train_agent(
        agent_type=agent_type,
        reward_fn=sparse_fn,
        opponent_policy_fn=opponent_fn,
        n_episodes=n_episodes,
        log_interval=n_episodes + 1,
        eval_interval=n_episodes + 1,
        agent_config=hparams,
        run_name=f"hs_{agent_type}",
        verbose=False,
    )
    return {"agent": agent_type, **hparams, "final_win_rate": metrics["final_win_rate"]}


def _make_fine_values(best, coarse_values, n_fine, zoom):
    sorted_c = sorted(set(coarse_values))
    if len(sorted_c) < 2:
        return list(sorted_c)
    avg_step = (sorted_c[-1] - sorted_c[0]) / (len(sorted_c) - 1)
    fine_step = avg_step / zoom
    half = (n_fine - 1) / 2
    lo, hi = sorted_c[0], sorted_c[-1]
    vals = sorted(set(
        round(max(lo, min(hi, best + fine_step * (i - half))), 10)
        for i in range(n_fine)
    ))
    return vals


def run_hypersearch(optimal_pi, opponent_policy_fn, agent_types, n_episodes,
                    n_parallel=None, output_csv="hypersearch_results.csv"):
    """Parallel grid search over hyperparameters using the sparse reward."""
    import itertools
    from concurrent.futures import ProcessPoolExecutor, as_completed

    _n_parallel = n_parallel or config.HYPERSEARCH_PARALLEL
    grid = config.HYPERSEARCH_GRID
    mc_keys = ["epsilon_0", "decay_rate", "optimistic_init"]
    ql_keys = ["epsilon_0", "decay_rate", "optimistic_init", "alpha_0", "decay_control"]

    all_rows = []
    completed_keys = set()

    if output_csv and os.path.exists(output_csv):
        with open(output_csv, newline="") as f:
            for row in csv.DictReader(f):
                all_rows.append(row)
                keys_for_agent = mc_keys if row["agent"] == "mc" else ql_keys
                key = (row["agent"],) + tuple(str(row.get(k, "")) for k in keys_for_agent)
                completed_keys.add(key)
        print(f"  Checkpoint: {len(all_rows)} results already completed.")

    csv_cols = None

    for agent_type in agent_types:
        keys = mc_keys if agent_type == "mc" else ql_keys
        combinations = list(itertools.product(*[grid[k] for k in keys]))

        pending = [v for v in combinations
                   if (agent_type,) + tuple(str(x) for x in v) not in completed_keys]
        total = len(combinations)

        print(f"\n{'=' * 60}")
        print(f"HYPERSEARCH: {agent_type.upper()} — {len(pending)} remaining / {total} total")
        print(f"{'=' * 60}")

        if not pending:
            continue

        worker_args = [(agent_type, keys, values, optimal_pi, n_episodes)
                       for values in pending]

        n_completed = total - len(pending)
        with ProcessPoolExecutor(max_workers=_n_parallel) as executor:
            futures = {executor.submit(_hs_worker, arg): arg for arg in worker_args}
            for future in as_completed(futures):
                row = future.result()
                all_rows.append(row)
                n_completed += 1
                hstr = " | ".join(f"{k}={row[k]}" for k in keys)
                print(f"  [{n_completed:>3}/{total}] WR={row['final_win_rate']:.4f} | {hstr}")

                if output_csv:
                    if csv_cols is None:
                        csv_cols = list(row.keys())
                    file_exists = os.path.exists(output_csv)
                    with open(output_csv, "a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=csv_cols)
                        if not file_exists:
                            writer.writeheader()
                        writer.writerow(row)

    sortable = [r for r in all_rows
                if str(r.get("final_win_rate", "")).replace(".", "").replace("-", "").isdigit()]
    sortable.sort(key=lambda r: float(r["final_win_rate"]), reverse=True)
    print(f"\n{'=' * 60}\nHYPERSEARCH TOP 20\n{'=' * 60}")
    for r in sortable[:20]:
        keys_for_agent = mc_keys if r["agent"] == "mc" else ql_keys
        hstr = " | ".join(f"{k}={r[k]}" for k in keys_for_agent)
        print(f"  [{r['agent'].upper()}] WR={float(r['final_win_rate']):.4f}  {hstr}")

    if all_rows and output_csv:
        cols = list(all_rows[0].keys())
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nResults saved to: {output_csv}")

    return all_rows


def run_hypersearch_zoom(optimal_pi, opponent_policy_fn, agent_types,
                         coarse_episodes=None, fine_episodes=None,
                         output_csv="hypersearch_results.csv"):
    """Coarse-to-fine hyperparameter grid search."""
    import itertools
    from concurrent.futures import ProcessPoolExecutor, as_completed

    coarse_episodes = coarse_episodes or config.HYPERSEARCH_COARSE_EPISODES
    fine_episodes = fine_episodes or config.HYPERSEARCH_FINE_EPISODES

    grid = config.HYPERSEARCH_GRID
    n_fine = config.HYPERSEARCH_FINE_N
    zoom = config.HYPERSEARCH_FINE_ZOOM
    mc_keys = ["epsilon_0", "decay_rate", "optimistic_init"]
    ql_keys = ["epsilon_0", "decay_rate", "optimistic_init", "alpha_0", "decay_control"]

    print(f"\n{'=' * 60}")
    print(f"ZOOM HYPERSEARCH — Phase 1: Coarse ({coarse_episodes:,} eps each)")
    print(f"{'=' * 60}")

    coarse_csv = output_csv.replace(".csv", "_coarse.csv") if output_csv else None
    coarse_rows = run_hypersearch(
        optimal_pi, opponent_policy_fn, agent_types, coarse_episodes,
        output_csv=coarse_csv,
    )

    fine_rows = []
    fine_csv = output_csv.replace(".csv", "_fine.csv") if output_csv else None
    fine_cols = None
    _n_parallel = config.HYPERSEARCH_PARALLEL

    for agent_type in agent_types:
        keys = mc_keys if agent_type == "mc" else ql_keys
        agent_coarse = [r for r in coarse_rows if r["agent"] == agent_type]
        best = max(agent_coarse, key=lambda r: float(r["final_win_rate"]))
        fine_grid = {k: _make_fine_values(float(best[k]), grid[k], n_fine, zoom)
                     for k in keys}
        combinations = list(itertools.product(*[fine_grid[k] for k in keys]))

        fine_done_keys = set()
        if fine_csv and os.path.exists(fine_csv):
            with open(fine_csv, newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("agent") == agent_type:
                        fine_rows.append(row)
                        fine_done_keys.add(tuple(str(row.get(k, "")) for k in keys))

        pending = [v for v in combinations
                   if tuple(str(x) for x in v) not in fine_done_keys]
        total = len(combinations)

        print(f"\n{'=' * 60}")
        print(f"ZOOM Phase 2: Fine {agent_type.upper()} "
              f"({len(pending)} remaining / {total}, {fine_episodes:,} eps each)")
        print(f"  Coarse best: WR={float(best['final_win_rate']):.4f}")
        print(f"{'=' * 60}")

        if not pending:
            continue

        worker_args = [(agent_type, keys, values, optimal_pi, fine_episodes)
                       for values in pending]
        n_completed = total - len(pending)
        with ProcessPoolExecutor(max_workers=_n_parallel) as executor:
            futures = {executor.submit(_hs_worker, arg): arg for arg in worker_args}
            for future in as_completed(futures):
                row = future.result()
                fine_rows.append(row)
                n_completed += 1
                if fine_csv:
                    if fine_cols is None:
                        fine_cols = list(row.keys())
                    file_exists = os.path.exists(fine_csv)
                    with open(fine_csv, "a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fine_cols)
                        if not file_exists:
                            writer.writeheader()
                        writer.writerow(row)
                hstr = " | ".join(f"{k}={row[k]}" for k in keys)
                print(f"  [{n_completed:>3}/{total}] WR={row['final_win_rate']:.4f} | {hstr}")

    all_rows = sorted(coarse_rows + fine_rows,
                      key=lambda r: r["final_win_rate"], reverse=True)

    print(f"\n{'=' * 60}\nZOOM HYPERSEARCH TOP 10\n{'=' * 60}")
    for r in all_rows[:10]:
        hstr = " | ".join(
            f"{k}={r[k]}" for k in r
            if k not in ("agent", "phase", "final_win_rate")
        )
        print(f"  [{r['agent'].upper()}] WR={r['final_win_rate']:.4f}  {hstr}")

    if all_rows and output_csv:
        all_cols = list(dict.fromkeys(k for r in all_rows for k in r))
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_cols)
            writer.writeheader()
            for r in all_rows:
                writer.writerow({c: r.get(c, "") for c in all_cols})
        print(f"\nResults saved to: {output_csv}")

    return all_rows


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Pig Dice RL reward design experiments")

    parser.add_argument("--run-vi", action="store_true")
    parser.add_argument("--verify-pbrs", action="store_true")
    parser.add_argument("--run-irl", action="store_true")
    parser.add_argument("--run-st-pete", action="store_true")
    parser.add_argument("--density", action="store_true")
    parser.add_argument("--density-episodes", type=int, default=5_000)
    parser.add_argument("--plot", action="store_true")

    parser.add_argument("--reward", type=str, default=None)
    parser.add_argument("--theta-key", type=str, default=None)
    parser.add_argument("--agent", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--n-workers", type=int, default=None,
                        help="Parallel workers (use 2 for Colab/limited RAM)")

    parser.add_argument("--run-hypersearch", action="store_true")
    parser.add_argument("--hypersearch-zoom", action="store_true")
    parser.add_argument("--hypersearch-agent", type=str, default=None)
    parser.add_argument("--hypersearch-parallel", type=int, default=None)
    parser.add_argument("--hypersearch-coarse-episodes", type=int, default=None)
    parser.add_argument("--hypersearch-fine-episodes", type=int, default=None)
    parser.add_argument("--hypersearch-csv", type=str, default="hypersearch_results.csv")

    parser.add_argument("--train-best", action="store_true")
    parser.add_argument("--curves-csv", type=str, default="learning_curves.csv")
    parser.add_argument("--output-pdf", type=str, default="learning_curves.pdf")
    parser.add_argument("--target-wr", type=float, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--train-seeds", type=int, default=None)

    parser.add_argument("--results-csv", type=str, default=None)
    parser.add_argument("--baseline-csv", type=str, default=None)

    args = parser.parse_args()

    if args.hypersearch_parallel is not None:
        config.HYPERSEARCH_PARALLEL = args.hypersearch_parallel

    if args.plot:
        plot_from_csv(args.curves_csv, args.output_pdf, target_wr=args.target_wr)
        return

    if args.run_st_pete:
        from st_petersburg import plot_static_wealth, plot_cumulative_wealth
        print("\n" + "=" * 60)
        print("ST. PETERSBURG PARADOX EXPERIMENTS")
        print("=" * 60)
        print("\n--- Static Wealth (10M games, w=10) ---")
        plot_static_wealth("utility_consistency_check.pdf")
        print("\n--- Cumulative Wealth (1K games, w=10) ---")
        plot_cumulative_wealth("st_petersburg_wealth_decay.pdf")
        if not any([args.run_vi, args.verify_pbrs, args.run_irl,
                    args.reward, args.agent, args.run_hypersearch,
                    args.hypersearch_zoom, args.train_best, args.density]):
            return

    if args.density:
        sample_reward_density(n_episodes=args.density_episodes)
        return

    print("\n" + "=" * 60)
    print("COMPUTING OPTIMAL POLICY")
    print("=" * 60)
    V_arr, optimal_pi = compute_or_load_optimal_policy(verbose=True)

    if args.run_vi:
        print("Value Iteration complete.")
        return

    opponent_fn = make_optimal_policy_fn(optimal_pi)

    if args.verify_pbrs:
        verify_pbrs_policy_invariance(optimal_pi, verbose=True)
        return

    if args.run_irl:
        run_irl_experiments(optimal_pi, opponent_fn)
        return

    hs_agents = (
        [args.hypersearch_agent] if args.hypersearch_agent in ("mc", "ql")
        else ["mc", "ql"]
    )

    if args.run_hypersearch:
        n_eps = args.episodes or config.HYPERSEARCH_EPISODES
        run_hypersearch(optimal_pi, opponent_fn, hs_agents, n_eps,
                        output_csv=args.hypersearch_csv)
        return

    if args.hypersearch_zoom:
        coarse_eps = args.hypersearch_coarse_episodes or config.HYPERSEARCH_COARSE_EPISODES
        fine_eps = args.hypersearch_fine_episodes or config.HYPERSEARCH_FINE_EPISODES
        run_hypersearch_zoom(optimal_pi, opponent_fn, hs_agents,
                             coarse_episodes=coarse_eps, fine_episodes=fine_eps,
                             output_csv=args.hypersearch_csv)
        return

    if args.train_best:
        if not os.path.exists(args.hypersearch_csv):
            print(f"ERROR: {args.hypersearch_csv} not found. Run hypersearch first.")
            sys.exit(1)
        run_best_training(
            hypersearch_csv=args.hypersearch_csv,
            optimal_pi=optimal_pi,
            opponent_fn=opponent_fn,
            n_seeds=args.train_seeds,
            max_episodes=args.max_episodes,
            target_wr=args.target_wr,
            output_csv=args.curves_csv,
            output_pdf=args.output_pdf,
        )
        return

    # --- Reward Design Experiments ---
    print("\n" + "=" * 60)
    print("REWARD DESIGN EXPERIMENTS")
    print("=" * 60)

    n_seeds = args.seeds or config.N_SEEDS
    n_workers = args.n_workers if args.n_workers is not None else 2

    reward_methods = config.REWARD_METHODS
    if args.reward:
        reward_methods = [m for m in reward_methods if m["name"] == args.reward]
        if not reward_methods:
            print(f"Unknown reward: {args.reward}")
            sys.exit(1)
    if args.theta_key:
        reward_methods = [m for m in reward_methods if m.get("theta_key") == args.theta_key]
        if not reward_methods:
            print(f"Unknown theta_key: {args.theta_key}")
            sys.exit(1)

    agent_types = config.AGENT_TYPES
    if args.agent:
        agent_types = [args.agent]

    all_results = {}
    os.makedirs("results", exist_ok=True)

    for reward_cfg in reward_methods:
        reward_name = reward_cfg["name"]
        reward_kwargs = {k: v for k, v in reward_cfg.items() if k != "name"}

        for agent_type in agent_types:
            label = reward_kwargs.get("theta_key", reward_name)
            run_name = f"{label}_{agent_type}"
            run_csv = os.path.join("results", f"{run_name}.csv")

            if os.path.exists(run_csv):
                print(f"\n  SKIP {run_name} — {run_csv} already exists")
                continue

            n_eps = args.episodes or (
                config.TRAINING_EPISODES_MC if agent_type == "mc"
                else config.TRAINING_EPISODES_QL
            )

            print(f"\n{'─' * 50}")
            print(f"  Reward: {reward_name} | Agent: {agent_type.upper()} | "
                  f"{n_eps:,} eps | Seeds: {n_seeds}")
            print(f"{'─' * 50}")

            metrics = train_agent_multi_seed(
                agent_type=agent_type,
                reward_name=reward_name,
                reward_kwargs=reward_kwargs,
                optimal_pi=optimal_pi,
                n_episodes=n_eps,
                n_seeds=n_seeds,
                n_workers=n_workers,
                log_interval=1000,
                eval_interval=config.GREEDY_EVAL_INTERVAL,
                eval_episodes=config.GREEDY_EVAL_EPISODES,
                run_name=run_name,
            )
            all_results[run_name] = metrics
            save_results_csv({run_name: metrics}, run_csv)
            print(f"  Saved: {run_csv}")

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Method':<30} {'WR (mean +/- std)':>18} {'Density':>9}")
    print("─" * 70)
    for name, m in all_results.items():
        wr_str = f"{m['final_win_rate']:.4f} +/- {m.get('final_win_rate_std', 0):.4f}"
        density = m.get("mean_reward_density", float("nan"))
        print(f"{name:<30} {wr_str:>18} {density:>9.5f}")

    running_sparse_only = (
        len(reward_methods) == 1 and reward_methods[0]["name"] == "sparse"
    )
    results_csv = args.results_csv or (
        "sparse_baseline.csv" if running_sparse_only else "results.csv"
    )
    save_results_csv(all_results, results_csv)

    baseline = None
    baseline_path = args.baseline_csv
    if baseline_path is None and not running_sparse_only:
        if os.path.exists("sparse_baseline.csv"):
            baseline_path = "sparse_baseline.csv"
    if baseline_path and os.path.exists(baseline_path):
        baseline = load_results_csv(baseline_path)
        print(f"Loaded baseline from: {baseline_path}")

    plot_reward_design_curves(all_results, output_pdf="reward_design_curves.pdf",
                              baseline=baseline)


if __name__ == "__main__":
    main()
