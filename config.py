# ── Training ──────────────────────────────────────────────────────────
TRAINING_EPISODES_MC = 1_000_000
TRAINING_EPISODES_QL = 1_000_000
EVAL_EPISODES = 10_000
N_SEEDS = 4

GREEDY_EVAL_INTERVAL = 10_000
GREEDY_EVAL_EPISODES = 2_000

# ── Hyperparameter search ─────────────────────────────────────────────
HYPERSEARCH_EPISODES = 200_000
HYPERSEARCH_COARSE_EPISODES = 100_000
HYPERSEARCH_FINE_EPISODES = 200_000
HYPERSEARCH_PARALLEL = 4

HYPERSEARCH_GRID = {
    "epsilon_0":       [0.5, 0.75, 1.0],
    "decay_rate":      [0.999999, 0.9999995, 0.9999999],
    "optimistic_init": [0.5, 0.75, 1.0],
    # QL-only
    "alpha_0":         [0.1, 0.3, 0.5],
    "decay_control":   [100_000, 250_000, 500_000],
}

HYPERSEARCH_FINE_N = 3
HYPERSEARCH_FINE_ZOOM = 4

# ── Best-hyperparameter training (--train-best) ──────────────────────
TRAIN_N_SEEDS = 4
TRAIN_MAX_EPISODES = 2_000_000
TRAIN_GREEDY_INTERVAL = 50_000
TRAIN_GREEDY_EVAL_EPS = 2_000
TARGET_WIN_RATE = 0.45

# ── MC Control (best from hypersearch) ────────────────────────────────
MC_CONFIG = {
    "optimistic_init": 0.5,
    "epsilon_0": 0.5625,
    "decay_rate": 0.9999999,
    "gamma": 1.0,
}

# ── Q-Learning (best from hypersearch) ────────────────────────────────
QL_CONFIG = {
    "optimistic_init": 0.75,
    "epsilon_0": 0.5,
    "decay_rate": 0.9999999,
    "alpha_0": 0.5,
    "decay_control": 500_000,
    "gamma": 1.0,
}

# ── Intrinsic Motivation ──────────────────────────────────────────────
INTRINSIC_ETA = 0.1

# ── IRL ───────────────────────────────────────────────────────────────
IRL_CONFIG = {
    "alpha": 0.01,
    "alpha_decay": 0.05,
    "max_outer_iters": 300,
    "tol": 1e-4,
    "n_eval_episodes": 2000,
    "n_expert_trajectories": [200, 2000],
    "expert_types": ["optimal", "sub_optimal"],
}

# ── Reward methods to run ─────────────────────────────────────────────
REWARD_METHODS = [
    {"name": "sparse"},
    {"name": "pbrs", "gamma": 1.0},
    {"name": "pbrs2", "gamma": 1.0},
    {"name": "naive_curiosity", "eta": INTRINSIC_ETA},
    {"name": "icm", "eta": INTRINSIC_ETA},
    {"name": "count_based", "eta": INTRINSIC_ETA},
    {"name": "irl", "theta_key": "irl_optimal_N200"},
    {"name": "irl", "theta_key": "irl_optimal_N2000"},
    {"name": "irl", "theta_key": "irl_sub_optimal_N200"},
    {"name": "irl", "theta_key": "irl_sub_optimal_N2000"},
]

AGENT_TYPES = ["mc", "ql"]
