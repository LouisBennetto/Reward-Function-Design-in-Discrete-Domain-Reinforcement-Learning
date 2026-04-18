"""
Plot learning curves from saved CSVs — one PDF per reward method.

Each figure shows the sparse baseline (faded, dotted) alongside the
method's MC (red) and QL (blue) curves with +/-1 std shading.

Usage:
    python plot_results.py                        # default: results/ dir
    python plot_results.py --results-dir ./data
"""

import argparse
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = [
    ("PBRS",                    "pbrs_mc.csv",                    "pbrs_ql.csv"),
    ("PBRS-2",                  "pbrs2_mc.csv",                   "pbrs2_ql.csv"),
    ("Naive Curiosity",         "naive_curiosity_mc.csv",         "naive_curiosity_ql.csv"),
    ("Count-Based",             "count_based_mc.csv",             "count_based_ql.csv"),
    ("ICM",                     "icm_mc.csv",                     "icm_ql.csv"),
    ("IRL Optimal N=200",       "irl_optimal_N200_mc.csv",        "irl_optimal_N200_ql.csv"),
    ("IRL Optimal N=2000",      "irl_optimal_N2000_mc.csv",       "irl_optimal_N2000_ql.csv"),
    ("IRL Sub-Optimal N=200",   "irl_sub_optimal_N200_mc.csv",    "irl_sub_optimal_N200_ql.csv"),
    ("IRL Sub-Optimal N=2000",  "irl_sub_optimal_N2000_mc.csv",   "irl_sub_optimal_N2000_ql.csv"),
]

MC_COLOR = "#d62728"
QL_COLOR = "#1f77b4"
BASE_ALPHA = 0.11
BASE_LINE_ALPHA = 0.45
BAND_ALPHA = 0.18
LW = 1.8
LW_BASE = 1.4

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load(path):
    return pd.read_csv(path).sort_values("episode")


def episodes_m(df):
    return df["episode"].values / 1_000_000


def plot_band(ax, df, col_mean, col_std, color, alpha_line, alpha_fill,
              lw, ls="-", label=None, zorder=2):
    x = episodes_m(df)
    mu = df[col_mean].values
    sd = df[col_std].values
    ax.plot(x, mu, color=color, lw=lw, ls=ls, label=label,
            alpha=alpha_line, zorder=zorder)
    ax.fill_between(x, mu - sd, mu + sd, color=color,
                    alpha=alpha_fill, zorder=zorder - 1)


def make_figure(label, df_mc, df_ql, sparse_mc, sparse_ql, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.grid(which="major", color="#cccccc", linewidth=0.7, zorder=0)
    ax.grid(which="minor", color="#e8e8e8", linewidth=0.4, zorder=0)
    ax.minorticks_on()
    ax.set_axisbelow(True)

    plot_band(ax, sparse_mc, "mean_wr", "std_wr",
              MC_COLOR, BASE_LINE_ALPHA, BASE_ALPHA, LW_BASE,
              ls=":", label="Sparse MC", zorder=1)
    plot_band(ax, sparse_ql, "mean_wr", "std_wr",
              QL_COLOR, BASE_LINE_ALPHA, BASE_ALPHA, LW_BASE,
              ls=":", label="Sparse QL", zorder=1)

    plot_band(ax, df_mc, "mean_wr", "std_wr",
              MC_COLOR, 1.0, BAND_ALPHA, LW,
              label=f"{label} MC", zorder=3)
    plot_band(ax, df_ql, "mean_wr", "std_wr",
              QL_COLOR, 1.0, BAND_ALPHA, LW,
              label=f"{label} QL", zorder=3)

    ax.set_xlabel("Episodes (millions)")
    ax.set_ylabel("Win rate")
    ax.set_ylim(0, 0.5306)
    ax.set_xlim(0, episodes_m(df_mc).max())
    ax.legend(fontsize=8, framealpha=0.6)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def make_sparse_figure(sparse_mc, sparse_ql, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.grid(which="major", color="#cccccc", linewidth=0.7, zorder=0)
    ax.grid(which="minor", color="#e8e8e8", linewidth=0.4, zorder=0)
    ax.minorticks_on()
    ax.set_axisbelow(True)

    plot_band(ax, sparse_mc, "mean_wr", "std_wr",
              MC_COLOR, 1.0, BAND_ALPHA, LW, label="Sparse MC", zorder=3)
    plot_band(ax, sparse_ql, "mean_wr", "std_wr",
              QL_COLOR, 1.0, BAND_ALPHA, LW, label="Sparse QL", zorder=3)

    ax.set_xlabel("Episodes (millions)")
    ax.set_ylabel("Win rate")
    ax.set_ylim(0, 0.5306)
    ax.set_xlim(0, episodes_m(sparse_mc).max())
    ax.legend(fontsize=8, framealpha=0.6)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Pig Dice learning curves")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    results_dir = args.results_dir
    if not os.path.isdir(results_dir):
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    baseline_path = os.path.join(results_dir, "sparse_baseline.csv")
    mc_path = os.path.join(results_dir, "sparse_mc.csv")
    ql_path = os.path.join(results_dir, "sparse_ql.csv")

    if os.path.exists(baseline_path):
        sparse_all = load(baseline_path)
        sparse_mc = sparse_all[sparse_all["run_name"] == "sparse_mc"].copy()
        sparse_ql = sparse_all[sparse_all["run_name"] == "sparse_ql"].copy()
    elif os.path.exists(mc_path) and os.path.exists(ql_path):
        sparse_mc = load(mc_path)
        sparse_ql = load(ql_path)
    else:
        print(f"Sparse baseline not found in: {results_dir}")
        sys.exit(1)

    make_sparse_figure(sparse_mc, sparse_ql,
                       os.path.join(results_dir, "curves_sparse.pdf"))

    for label, mc_file, ql_file in METHODS:
        mc_path = os.path.join(results_dir, mc_file)
        ql_path = os.path.join(results_dir, ql_file)
        if not os.path.exists(mc_path) or not os.path.exists(ql_path):
            print(f"  SKIP {label} — CSV not found")
            continue

        df_mc = load(mc_path)
        df_ql = load(ql_path)

        slug = label.lower().replace(" ", "_").replace("-", "")
        out_path = os.path.join(results_dir, f"curves_{slug}.pdf")
        make_figure(label, df_mc, df_ql, sparse_mc, sparse_ql, out_path)

    print("Done.")


if __name__ == "__main__":
    main()
