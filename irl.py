"""
Maximum Entropy IRL for Pig Dice.

The opponent's s_B is bucketed to {0, 8, ..., 96} internally (EXP_OPP=8
approximation) to make Bellman arrays ~8x smaller. The recovered policy
is then expanded to all s_B in 0–99.
"""

import numpy as np

TARGET = 100
ROLL = 0
HOLD = 1
MAX_SCORE = 106
MAX_KAPPA = 106

_EXP_OPP = 8
_SB_STEP = _EXP_OPP
_SB_VALUES = np.arange(0, TARGET, _SB_STEP, dtype=np.int32)
_N_SB = len(_SB_VALUES)
_SA = np.arange(TARGET, dtype=np.int32)


def phi(s_A, s_B, kappa, action):
    """
    Feature vector in R^3:
        phi_1 = (1{ROLL} - 1{HOLD}) * (20 - kappa) / 6
        phi_2 = kappa / max(1, 100 - s_A)
        phi_3 = 1{ROLL} * 1{s_B >= 85}
    """
    sign = 1.0 if action == ROLL else -1.0
    features = np.zeros(3, dtype=np.float64)
    features[0] = sign * (20.0 - kappa) / 6.0
    features[1] = float(kappa) / max(1, TARGET - s_A)
    features[2] = 1.0 if (action == ROLL and s_B >= 85) else 0.0
    return features


def compute_feature_counts_from_trajectories(trajectories):
    N = len(trajectories)
    mu_E = np.zeros(3, dtype=np.float64)
    for traj in trajectories:
        for (s_A, s_B, kappa, action) in traj:
            mu_E += phi(s_A, s_B, kappa, action)
    mu_E /= N
    return mu_E


def reward_theta(s_A, s_B, kappa, action, theta):
    return np.dot(theta, phi(s_A, s_B, kappa, action))


def solve_policy_for_theta(theta, gamma=1.0, tol=1e-5, max_iters=200,
                           V_init=None, **_kwargs):
    """Vectorised VI under R_theta. Returns (pi_arr [100,100,106], V)."""
    MAX_K = MAX_KAPPA
    N_SB = _N_SB

    SA = _SA[:, None, None]
    B = np.arange(N_SB, dtype=np.int32)[None, :, None]
    K = np.arange(MAX_K, dtype=np.int32)[None, None, :]
    SB_actual = _SB_VALUES[None, :, None].astype(np.float64)

    terminal = (SA + K >= TARGET)

    _phi1 = (20.0 - K.astype(np.float64)) / 6.0
    _denom = np.maximum(TARGET - SA.astype(np.float64), 1.0)
    _phi2 = K.astype(np.float64) / _denom
    _opp_high = (SB_actual >= 85.0)

    R_hold = theta[0] * (-_phi1) + theta[1] * _phi2
    R_roll = theta[0] * _phi1 + theta[1] * _phi2 + theta[2] * _opp_high

    die_data = []
    for die in range(2, 7):
        kd = MAX_K - die
        new_k = K[:, :, :kd] + die
        wins = (SA + new_k >= TARGET)
        nk_f = new_k.astype(np.float64)
        r_win = (theta[0] * (20.0 - nk_f) / 6.0
                 + theta[1] * nk_f / _denom
                 + theta[2] * _opp_high)
        die_data.append((die, kd, wins, r_win))

    V = np.zeros((MAX_SCORE, N_SB, MAX_K), dtype=np.float64)

    pi_compact = None
    for _ in range(max_iters):
        V_2d = V[:MAX_SCORE, :N_SB, 0]

        post_hold = np.zeros((TARGET, N_SB, MAX_K), dtype=np.float64)
        for ki in range(TARGET):
            max_i = TARGET - ki
            if max_i <= 0:
                break
            post_hold[:max_i, :N_SB - 1, ki] = V_2d[ki:ki + max_i, 1:N_SB]

        v_hold = np.where(terminal, R_hold, R_hold + gamma * post_hold)

        Vb = np.zeros((TARGET, N_SB), dtype=np.float64)
        Vb[:, :N_SB - 1] = V[:TARGET, 1:N_SB, 0]
        v_bust = Vb[:, :, np.newaxis]

        v_continue = np.zeros((TARGET, N_SB, MAX_K), dtype=np.float64)
        for die, kd, wins, r_win in die_data:
            v_slice = gamma * V[:TARGET, :N_SB, die:MAX_K]
            v_continue[:, :, :kd] += np.where(wins, r_win, v_slice)

        v_roll = R_roll + gamma * ((1.0 / 6.0) * v_bust
                                   + (1.0 / 6.0) * v_continue)

        V[:TARGET, :N_SB, :MAX_K] = np.where(
            terminal, R_hold, np.maximum(v_hold, v_roll)
        )

        pi_new = np.where(
            terminal | (v_hold >= v_roll), HOLD, ROLL
        ).astype(np.int8)
        if pi_compact is not None and np.array_equal(pi_new, pi_compact):
            break
        pi_compact = pi_new

    pi_arr = np.empty((TARGET, TARGET, MAX_K), dtype=np.int8)
    for j in range(TARGET):
        b = min(j // _SB_STEP, N_SB - 1)
        pi_arr[:, j, :] = pi_compact[:, b, :]

    return pi_arr, V


def compute_expected_feature_counts(pi, gamma=1.0, n_episodes=1000):
    """Expected feature counts under pi via MC rollout against hold-at-20."""
    from pig_dice_env import PigDiceEnv

    is_arr = isinstance(pi, np.ndarray)
    env = PigDiceEnv(opponent_policy=_hold_at_20_fn, agent_starts=True)
    mu = np.zeros(3, dtype=np.float64)

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            s_A, s_B, kappa = int(obs[0]), int(obs[1]), int(obs[2])
            if is_arr:
                action = (int(pi[s_A, s_B, kappa])
                          if s_A < TARGET and s_B < MAX_SCORE and kappa < MAX_KAPPA
                          else ROLL)
            else:
                action = pi.get((s_A, s_B, kappa), ROLL)
            mu += phi(s_A, s_B, kappa, action)
            obs, _, done, _, _ = env.step(action)

    return mu / n_episodes


def _hold_at_20_fn(my_banked, opp_banked, kappa):
    if my_banked + kappa >= TARGET:
        return HOLD
    return HOLD if kappa >= 20 else ROLL


def maxent_irl(expert_trajectories, n_features=3, alpha=0.01,
               alpha_decay=0.02, max_outer_iters=100, tol=1e-4,
               n_eval_episodes=1000, l2_reg=0.01, verbose=True):
    """
    Maximum Entropy IRL gradient descent.

    Returns:
        theta:   recovered weight vector
        history: list of per-iteration dicts
    """
    mu_E = compute_feature_counts_from_trajectories(expert_trajectories)
    if verbose:
        print(f"Expert mean feature counts: {mu_E}")

    theta = np.zeros(n_features, dtype=np.float64)
    history = []

    for outer_iter in range(max_outer_iters):
        pi_arr, _ = solve_policy_for_theta(theta)
        mu_theta = compute_expected_feature_counts(pi_arr,
                                                   n_episodes=n_eval_episodes)

        alpha_t = alpha / (1.0 + alpha_decay * outer_iter)
        grad = mu_theta - mu_E + l2_reg * theta
        theta -= alpha_t * grad

        grad_norm = np.linalg.norm(grad)
        history.append({
            "iter": outer_iter, "theta": theta.copy(),
            "grad_norm": grad_norm, "mu_E": mu_E.copy(),
            "mu_theta": mu_theta.copy(), "alpha_t": alpha_t,
        })

        if verbose:
            print(f"  IRL iter {outer_iter}: alpha={alpha_t:.5f}, "
                  f"||grad||={grad_norm:.6f}, theta={theta}")

        if grad_norm < tol:
            if verbose:
                print(f"  Converged at iteration {outer_iter}")
            break

    return theta, history


def generate_expert_trajectories(policy_fn, n_trajectories=1000,
                                 opponent_policy=None, agent_starts=True,
                                 epsilon=0.0):
    """Roll out policy_fn for n_trajectories episodes; epsilon = random-action probability."""
    from pig_dice_env import PigDiceEnv
    env = PigDiceEnv(
        opponent_policy=opponent_policy or _hold_at_20_fn,
        agent_starts=agent_starts,
    )
    trajectories = []
    for _ in range(n_trajectories):
        obs, _ = env.reset()
        traj = []
        done = False
        while not done:
            s_A, s_B, kappa = int(obs[0]), int(obs[1]), int(obs[2])
            action = (np.random.randint(2) if np.random.random() < epsilon
                      else policy_fn(s_A, s_B, kappa))
            traj.append((s_A, s_B, kappa, action))
            obs, _, done, _, _ = env.step(action)
        trajectories.append(traj)
    return trajectories
