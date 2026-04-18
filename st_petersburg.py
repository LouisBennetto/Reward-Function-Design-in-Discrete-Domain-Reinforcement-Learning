"""
St. Petersburg Paradox — Utility vs Raw Reward experiments.

Produces three PDFs:
    raw_reward.pdf               Raw reward (5M games, diverging)
    utility_static_wealth.pdf    Log utility, static wealth (5M games, stable)
    utility_cumulative_wealth.pdf  Log utility, cumulative wealth (1K games)

Usage:
    python st_petersburg.py
    python st_petersburg.py --output-dir results/
"""

import argparse
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW_COLORS  = ["#d62728", "#ff7f0e", "#e377c2"]
UTIL_COLORS = ["#1f3080", "#17becf", "#2ca02c"]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def play_st_petersburg(rng):
    payout = 1
    while rng.random() < 0.5:
        payout *= 2
    return payout


def log_utility(w, reward):
    return np.log(w + reward) - np.log(w)


def run_static_wealth(n_games=5_000_000, initial_wealth=10.0, seed=0,
                      log_interval=1000):
    rng = np.random.default_rng(seed)
    raw_sum = 0.0
    util_sum = 0.0

    n_points = n_games // log_interval
    games = np.empty(n_points, dtype=np.int64)
    mean_raw = np.empty(n_points, dtype=np.float64)
    mean_util = np.empty(n_points, dtype=np.float64)

    idx = 0
    for i in range(1, n_games + 1):
        payout = play_st_petersburg(rng)
        raw_sum += payout
        util_sum += log_utility(initial_wealth, payout)

        if i % log_interval == 0:
            games[idx] = i
            mean_raw[idx] = raw_sum / i
            mean_util[idx] = util_sum / i
            idx += 1

    return games, mean_raw, mean_util


def run_cumulative_wealth(n_games=1000, initial_wealth=10.0, seed=0):
    rng = np.random.default_rng(seed)
    w = initial_wealth
    raw_sum = 0.0
    util_sum = 0.0

    games = np.arange(1, n_games + 1)
    mean_raw = np.empty(n_games, dtype=np.float64)
    mean_util = np.empty(n_games, dtype=np.float64)

    for i in range(n_games):
        payout = play_st_petersburg(rng)
        raw_sum += payout
        util_sum += log_utility(w, payout)
        w += payout

        mean_raw[i] = raw_sum / (i + 1)
        mean_util[i] = util_sum / (i + 1)

    return games, mean_raw, mean_util


def plot_raw_reward(output_path, n_games=5_000_000, initial_wealth=10.0,
                   n_runs=3, log_interval=1000):
    seeds = [42, 123, 7]
    fig, ax = plt.subplots(figsize=(10, 6))

    for r in range(n_runs):
        print(f"  Raw Reward — Run {r+1}/{n_runs} (seed={seeds[r]}, {n_games:,} games)...")
        games, mean_raw, _ = run_static_wealth(
            n_games=n_games, initial_wealth=initial_wealth,
            seed=seeds[r], log_interval=log_interval,
        )
        ax.plot(games, mean_raw, color=RAW_COLORS[r], lw=1.5,
                label=f"Run {r+1}", alpha=0.85)

    ax.set_title("Raw Reward", fontsize=14, fontweight="bold")
    ax.set_xlabel("Games Played", fontsize=12)
    ax.set_ylabel("Average Winnings ($)", fontsize=12)
    ax.set_ylim(0, 50)
    ax.legend(fontsize=10, framealpha=0.6)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_utility_static_wealth(output_path, n_games=5_000_000, initial_wealth=10.0,
                               n_runs=3, log_interval=1000):
    seeds = [42, 123, 7]
    fig, ax = plt.subplots(figsize=(10, 6))

    for r in range(n_runs):
        print(f"  Static Wealth Utility — Run {r+1}/{n_runs} (seed={seeds[r]}, {n_games:,} games)...")
        games, _, mean_util = run_static_wealth(
            n_games=n_games, initial_wealth=initial_wealth,
            seed=seeds[r], log_interval=log_interval,
        )
        ax.plot(games, mean_util, color=UTIL_COLORS[r], lw=1.5,
                label=f"Run {r+1}", alpha=0.85)

    ax.set_title("Log Utility Transformed Reward (Static Wealth)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Games Played", fontsize=12)
    ax.set_ylabel("Average Utility Units", fontsize=12)
    ax.legend(fontsize=10, framealpha=0.6)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_utility_cumulative_wealth(output_path, n_games=1000, initial_wealth=10.0,
                                   n_runs=3):
    seeds = [42, 123, 7]
    fig, ax = plt.subplots(figsize=(10, 6))

    for r in range(n_runs):
        print(f"  Cumulative Wealth Utility — Run {r+1}/{n_runs} (seed={seeds[r]}, {n_games:,} games)...")
        games, _, mean_util = run_cumulative_wealth(
            n_games=n_games, initial_wealth=initial_wealth, seed=seeds[r],
        )
        ax.plot(games, mean_util, color=UTIL_COLORS[r], lw=1.5,
                label=f"Run {r+1}", alpha=0.85)

    ax.set_title("Log Utility Transformed Reward (Cumulative Wealth)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Games Played", fontsize=12)
    ax.set_ylabel("Average Marginal Utility", fontsize=12)
    ax.legend(fontsize=10, framealpha=0.6)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="St. Petersburg Paradox — Utility vs Raw Reward")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory for output PDFs (default: current dir)")
    parser.add_argument("--static-games", type=int, default=5_000_000)
    parser.add_argument("--cumul-games", type=int, default=1_000)
    parser.add_argument("--wealth", type=float, default=10.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("ST. PETERSBURG PARADOX EXPERIMENTS")
    print("=" * 60)

    print(f"\n--- Raw Reward ({args.static_games:,} games, w={args.wealth}) ---")
    plot_raw_reward(
        output_path=os.path.join(args.output_dir, "raw_reward.pdf"),
        n_games=args.static_games,
        initial_wealth=args.wealth,
    )

    print(f"\n--- Log Utility Static Wealth ({args.static_games:,} games, w={args.wealth}) ---")
    plot_utility_static_wealth(
        output_path=os.path.join(args.output_dir, "utility_static_wealth.pdf"),
        n_games=args.static_games,
        initial_wealth=args.wealth,
    )

    print(f"\n--- Log Utility Cumulative Wealth ({args.cumul_games:,} games, w={args.wealth}) ---")
    plot_utility_cumulative_wealth(
        output_path=os.path.join(args.output_dir, "utility_cumulative_wealth.pdf"),
        n_games=args.cumul_games,
        initial_wealth=args.wealth,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
