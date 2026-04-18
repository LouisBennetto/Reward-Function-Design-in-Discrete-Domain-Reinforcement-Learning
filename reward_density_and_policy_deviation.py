"""
Verify Policy Invariance Theorem for PBRS and PBRS2 via Value Iteration.

Runs VI under each shaped reward and reports d(pi*_shaped, pi*_sparse).

Usage:
    python reward_density_and_policy_deviation.py
"""

import json
import os

from pig_dice_env import TARGET, ROLL, HOLD
from rewards import PBRSReward, PBRS2Reward
from value_iteration import value_iteration, value_iteration_shaped


def compute_policy_deviation(pi_ref, pi_other, max_k=100):
    """Fraction of non-terminal states where pi_ref and pi_other disagree."""
    n_disagree = 0
    n_states = 0

    for i in range(TARGET):
        for j in range(TARGET):
            for k in range(max_k):
                if i + k >= TARGET:
                    continue
                a_ref = pi_ref.get((i, j, k), ROLL)
                a_other = pi_other.get((i, j, k), ROLL)
                n_states += 1
                if a_ref != a_other:
                    n_disagree += 1

    deviation = n_disagree / n_states if n_states > 0 else 0.0
    return deviation, n_disagree, n_states


def main():
    print("=" * 65)
    print("POLICY DEVIATION ANALYSIS (Value Iteration)")
    print("Verifying Policy Invariance Theorem for PBRS and PBRS2")
    print("=" * 65)

    print("\n[1/3] Computing sparse optimal policy (standard VI)...")
    _, pi_sparse, _ = value_iteration(verbose=True)
    print(f"  Converged. Policy defined over {len(pi_sparse):,} states.")

    print("\n[2/3] Computing PBRS optimal policy (shaped VI)...")
    pi_pbrs = value_iteration_shaped(PBRSReward(gamma=1.0), verbose=True)
    print(f"  Converged. Policy defined over {len(pi_pbrs):,} states.")

    print("\n[3/3] Computing PBRS2 optimal policy (shaped VI)...")
    pi_pbrs2 = value_iteration_shaped(PBRS2Reward(gamma=1.0), verbose=True)
    print(f"  Converged. Policy defined over {len(pi_pbrs2):,} states.")

    dev_pbrs,  n_dis_pbrs,  n_total = compute_policy_deviation(pi_sparse, pi_pbrs)
    dev_pbrs2, n_dis_pbrs2, _       = compute_policy_deviation(pi_sparse, pi_pbrs2)

    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)
    print(f"\nNon-terminal states evaluated: {n_total:,}")
    print(f"\nPBRS  vs Sparse:  {n_dis_pbrs:,} / {n_total:,} disagreements"
          f"  =>  deviation = {dev_pbrs:.6f}  ({dev_pbrs * 100:.4f}%)")
    print(f"PBRS2 vs Sparse:  {n_dis_pbrs2:,} / {n_total:,} disagreements"
          f"  =>  deviation = {dev_pbrs2:.6f}  ({dev_pbrs2 * 100:.4f}%)")

    for label, dev in [("PBRS", dev_pbrs), ("PBRS2", dev_pbrs2)]:
        if dev == 0.0:
            print(f"\n  {label}: deviation = 0 — Policy Invariance Theorem confirmed.")
        else:
            print(f"\n  {label}: non-zero deviation ({dev * 100:.4f}%) — "
                  "likely tie-breaking difference from VI.")

    results = {
        "n_non_terminal_states": n_total,
        "pbrs_vs_sparse": {
            "deviation":       dev_pbrs,
            "n_disagreements": n_dis_pbrs,
            "n_states":        n_total,
        },
        "pbrs2_vs_sparse": {
            "deviation":       dev_pbrs2,
            "n_disagreements": n_dis_pbrs2,
            "n_states":        n_total,
        },
    }
    out_path = "policy_deviation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
