"""Value Iteration for Pig Dice using self-play symmetry (Neller & Presser 2004)."""

import numpy as np
import os

TARGET = 100
ROLL = 0
HOLD = 1


def value_iteration(gamma=1.0, tol=1e-6, max_iters=200, verbose=False):
    """
    Self-play value iteration.

    Returns:
        V_dict: {} (unused, kept for API compatibility)
        pi:     dict (s_A, s_B, kappa) -> action
        P:      numpy array [i, j, k] of win probabilities
    """
    MAX_K = 100
    P = np.zeros((TARGET, TARGET, MAX_K), dtype=np.float64)

    for iteration in range(max_iters):
        P_old = P.copy()

        for i in range(TARGET):
            for j in range(TARGET):
                for k in range(MAX_K):
                    if i + k >= TARGET:
                        P[i, j, k] = 1.0
                        continue

                    new_i = i + k
                    v_hold = 1.0 if new_i >= TARGET else 1.0 - P[j, new_i, 0]

                    v_bust = 1.0 - P[j, i, 0]
                    v_continue = 0.0
                    for d in range(2, 7):
                        new_k = k + d
                        if i + new_k >= TARGET:
                            v_continue += 1.0
                        elif new_k < MAX_K:
                            v_continue += P[i, j, new_k]
                        else:
                            v_continue += 1.0
                    v_roll = (1.0 / 6.0) * v_bust + (1.0 / 6.0) * v_continue

                    P[i, j, k] = max(v_hold, v_roll)

        delta = np.max(np.abs(P - P_old))
        if verbose and (iteration % 5 == 0 or delta < tol):
            print(f"  VI iteration {iteration}, delta={delta:.10f}")
        if delta < tol:
            if verbose:
                print(f"  Converged at iteration {iteration}")
            break

    pi = {}
    for i in range(TARGET):
        for j in range(TARGET):
            for k in range(MAX_K):
                if i + k >= TARGET:
                    pi[(i, j, k)] = HOLD
                    continue

                new_i = i + k
                v_hold = 1.0 if new_i >= TARGET else 1.0 - P[j, new_i, 0]

                v_bust = 1.0 - P[j, i, 0]
                v_continue = 0.0
                for d in range(2, 7):
                    new_k = k + d
                    if i + new_k >= TARGET:
                        v_continue += 1.0
                    elif new_k < MAX_K:
                        v_continue += P[i, j, new_k]
                    else:
                        v_continue += 1.0
                v_roll = (1.0 / 6.0) * v_bust + (1.0 / 6.0) * v_continue

                pi[(i, j, k)] = HOLD if v_hold >= v_roll else ROLL

    return {}, pi, P


def make_optimal_policy_fn(pi_dict):
    """Convert a policy dict to a callable for use as opponent_policy."""
    def policy_fn(my_banked, opp_banked, my_kappa):
        key = (my_banked, opp_banked, my_kappa)
        if key in pi_dict:
            return pi_dict[key]
        if my_banked + my_kappa >= TARGET:
            return HOLD
        return ROLL
    return policy_fn


def save_policy(pi, P, path="optimal_policy.npz"):
    pi_keys = np.array(list(pi.keys()), dtype=np.int32)
    pi_vals = np.array(list(pi.values()), dtype=np.int32)
    np.savez_compressed(path, P=P, pi_keys=pi_keys, pi_vals=pi_vals)


def load_policy(path="optimal_policy.npz"):
    data = np.load(path)
    P = data["P"]
    pi_keys = [tuple(k) for k in data["pi_keys"]]
    pi_vals = data["pi_vals"].tolist()
    pi = dict(zip(pi_keys, pi_vals))
    return pi, P


def value_iteration_shaped(reward_fn, gamma=1.0, tol=1e-6, max_iters=200,
                           verbose=False):
    """
    VI under a shaped reward — used to verify PBRS policy invariance.

    The opponent's turn is approximated as: value = -V(j, i+k, 0).
    """
    MAX_K = 100
    V = np.zeros((TARGET, TARGET, MAX_K), dtype=np.float64)

    for iteration in range(max_iters):
        V_old = V.copy()

        for i in range(TARGET):
            for j in range(TARGET):
                for k in range(MAX_K):
                    if i + k >= TARGET:
                        V[i, j, k] = 0.0
                        continue

                    s = (i, j, k)

                    new_i = i + k
                    if new_i >= TARGET:
                        s_next_hold = (new_i, j, 0)
                        r_hold = reward_fn(s, HOLD, s_next_hold, 1.0)
                        v_hold = r_hold
                    else:
                        s_next_hold = (new_i, j, 0)
                        r_hold = reward_fn(s, HOLD, s_next_hold, 0.0)
                        v_hold = r_hold + gamma * (- V[j, new_i, 0])

                    s_next_bust = (i, j, 0)
                    r_bust = reward_fn(s, ROLL, s_next_bust, 0.0)
                    v_bust = r_bust + gamma * (- V[j, i, 0])

                    v_continue = 0.0
                    for d in range(2, 7):
                        new_k = k + d
                        if i + new_k >= TARGET:
                            s_next_d = (i, j, new_k)
                            r_d = reward_fn(s, ROLL, s_next_d, 1.0)
                            v_continue += r_d
                        elif new_k < MAX_K:
                            s_next_d = (i, j, new_k)
                            r_d = reward_fn(s, ROLL, s_next_d, 0.0)
                            v_continue += r_d + gamma * V[i, j, new_k]
                        else:
                            s_next_d = (i, j, new_k)
                            r_d = reward_fn(s, ROLL, s_next_d, 1.0)
                            v_continue += r_d

                    v_roll = (1.0 / 6.0) * v_bust + (1.0 / 6.0) * v_continue
                    V[i, j, k] = max(v_hold, v_roll)

        delta = np.max(np.abs(V - V_old))
        if verbose and (iteration % 5 == 0 or delta < tol):
            print(f"  VI-shaped iteration {iteration}, delta={delta:.10f}")
        if delta < tol:
            if verbose:
                print(f"  Converged at iteration {iteration}")
            break

    pi = {}
    for i in range(TARGET):
        for j in range(TARGET):
            for k in range(MAX_K):
                if i + k >= TARGET:
                    pi[(i, j, k)] = HOLD
                    continue

                s = (i, j, k)
                new_i = i + k
                if new_i >= TARGET:
                    s_next_hold = (new_i, j, 0)
                    r_hold = reward_fn(s, HOLD, s_next_hold, 1.0)
                    v_hold = r_hold
                else:
                    s_next_hold = (new_i, j, 0)
                    r_hold = reward_fn(s, HOLD, s_next_hold, 0.0)
                    v_hold = r_hold + gamma * (- V[j, new_i, 0])

                s_next_bust = (i, j, 0)
                r_bust = reward_fn(s, ROLL, s_next_bust, 0.0)
                v_bust = r_bust + gamma * (- V[j, i, 0])

                v_continue = 0.0
                for d in range(2, 7):
                    new_k = k + d
                    if i + new_k >= TARGET:
                        s_next_d = (i, j, new_k)
                        r_d = reward_fn(s, ROLL, s_next_d, 1.0)
                        v_continue += r_d
                    elif new_k < MAX_K:
                        s_next_d = (i, j, new_k)
                        r_d = reward_fn(s, ROLL, s_next_d, 0.0)
                        v_continue += r_d + gamma * V[i, j, new_k]
                    else:
                        s_next_d = (i, j, new_k)
                        r_d = reward_fn(s, ROLL, s_next_d, 1.0)
                        v_continue += r_d

                v_roll = (1.0 / 6.0) * v_bust + (1.0 / 6.0) * v_continue
                pi[(i, j, k)] = HOLD if v_hold >= v_roll else ROLL

    return pi


if __name__ == "__main__":
    print("Running Value Iteration for Pig Dice (self-play)...")
    _, pi, P = value_iteration(verbose=True)

    print(f"\nP(0,0,0) = {P[0, 0, 0]:.4f}  (expected ~0.5306)")
    print(f"P(50,0,0) = {P[50, 0, 0]:.4f}")
    print(f"P(0,50,0) = {P[0, 50, 0]:.4f}")
    print(f"Policy size: {len(pi)} states")

    print(f"\npi(0,0,20) = {'HOLD' if pi.get((0,0,20)) == HOLD else 'ROLL'}")
    print(f"pi(0,0,10) = {'HOLD' if pi.get((0,0,10)) == HOLD else 'ROLL'}")
    print(f"pi(90,0,10) = {'HOLD' if pi.get((90,0,10)) == HOLD else 'ROLL'} (should HOLD)")

    print("\nHold threshold at (0, 0, k):")
    for k in range(1, 40):
        if pi.get((0, 0, k), ROLL) == HOLD and pi.get((0, 0, k - 1), ROLL) == ROLL:
            print(f"  First HOLD at k={k}")
            break

    save_policy(pi, P)
    print("\nSaved to optimal_policy.npz")
