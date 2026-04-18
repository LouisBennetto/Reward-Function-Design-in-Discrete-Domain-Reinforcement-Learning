"""
Reward functions for Pig Dice experiments.

All callables have signature:
    reward_fn(s, a, s_next, env_reward) -> shaped_reward

where s = (s_A, s_B, kappa).
"""

import json
import os
import numpy as np

TARGET = 100
ROLL = 0
HOLD = 1


class SparseReward:
    name = "sparse"

    def __call__(self, s, a, s_next, env_reward, info=None):
        return env_reward


class PBRSReward:
    """PBRS with Phi_1(s) = 1{non-terminal} * (s_A - s_B) / 100."""
    name = "pbrs"

    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def _phi(self, state):
        s_A, s_B, kappa = state
        if s_A + kappa >= TARGET or s_B >= TARGET:
            return 0.0
        return (s_A - s_B) / TARGET

    def __call__(self, s, a, s_next, env_reward, info=None):
        F = self.gamma * self._phi(s_next) - self._phi(s)
        return env_reward + F


class PBRS2Reward:
    """PBRS with Phi_2(s) = 1{non-terminal} * kappa / 6."""
    name = "pbrs2"

    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def _phi(self, state):
        s_A, s_B, kappa = state
        if s_A + kappa >= TARGET or s_B >= TARGET:
            return 0.0
        return kappa / 6.0

    def __call__(self, s, a, s_next, env_reward, info=None):
        F = self.gamma * self._phi(s_next) - self._phi(s)
        return env_reward + F


class NaiveCuriosityReward:
    """Prediction-error bonus from a frequency-table successor-state model."""
    name = "naive_curiosity"

    def __init__(self, eta=0.1):
        self.eta = eta
        self.transition_counts = {}

    def _predict(self, s, a):
        key = (s, a)
        if key not in self.transition_counts:
            return None
        counts = self.transition_counts[key]
        return max(counts, key=counts.get)

    def _update(self, s, a, s_next):
        key = (s, a)
        if key not in self.transition_counts:
            self.transition_counts[key] = {}
        if s_next not in self.transition_counts[key]:
            self.transition_counts[key][s_next] = 0
        self.transition_counts[key][s_next] += 1

    def __call__(self, s, a, s_next, env_reward, info=None):
        prediction = self._predict(s, a)
        r_int = 1.0 if prediction != s_next else 0.0
        self._update(s, a, s_next)
        return env_reward + self.eta * r_int

    def reset_episode(self):
        pass


class ICMReward:
    """Tabular ICM: forward model predicts s_A' from (s, a), filtering die-roll noise."""
    name = "icm"

    def __init__(self, eta=0.1):
        self.eta = eta
        self.forward_counts = {}

    def _predict(self, s, a):
        key = (s, a)
        if key not in self.forward_counts:
            return None
        return max(self.forward_counts[key], key=self.forward_counts[key].get)

    def _update(self, s, a, s_A_next):
        key = (s, a)
        if key not in self.forward_counts:
            self.forward_counts[key] = {}
        if s_A_next not in self.forward_counts[key]:
            self.forward_counts[key][s_A_next] = 0
        self.forward_counts[key][s_A_next] += 1

    def __call__(self, s, a, s_next, env_reward, info=None):
        s = tuple(s) if not isinstance(s, tuple) else s
        s_A_next = s_next[0]
        prediction = self._predict(s, a)
        r_int = 1.0 if prediction != s_A_next else 0.0
        self._update(s, a, s_A_next)
        return env_reward + self.eta * r_int

    def reset_episode(self):
        pass


class CountBasedReward:
    """Visitation-count bonus: eta / sqrt(N(s,a) + 1), capped at N_MAX visits."""
    name = "count_based"
    N_MAX = 10_000

    def __init__(self, eta=0.1):
        self.eta = eta
        self.visit_counts = {}

    def __call__(self, s, a, s_next, env_reward, info=None):
        key = (s, a)
        count = self.visit_counts.get(key, 0)
        r_int = self.eta / np.sqrt(count + 1)
        if count < self.N_MAX:
            self.visit_counts[key] = count + 1
        return env_reward + r_int

    def reset_episode(self):
        pass


class IRLReward:
    """Linear reward R_theta(s,a) = theta^T phi(s,a) from MaxEnt IRL weights."""
    name = "irl"

    def __init__(self, theta_key, weights_file="precomputed_results/irl_weights.json"):
        if not os.path.exists(weights_file):
            raise FileNotFoundError(
                f"IRL weights not found at '{weights_file}'. "
                "Run 'python train.py --run-irl' first."
            )
        with open(weights_file) as f:
            all_weights = json.load(f)
        if theta_key not in all_weights:
            raise KeyError(
                f"Key '{theta_key}' not in {weights_file}. "
                f"Available: {list(all_weights.keys())}"
            )
        self.theta = np.array(all_weights[theta_key], dtype=np.float64)

    def _phi(self, s, a):
        s_A, s_B, kappa = s
        sign = 1.0 if a == ROLL else -1.0
        features = np.zeros(3, dtype=np.float64)
        features[0] = sign * (20.0 - kappa) / 6.0
        features[1] = float(kappa) / max(1, TARGET - s_A)
        features[2] = 1.0 if (a == ROLL and s_B >= 85) else 0.0
        return features

    def __call__(self, s, a, s_next, env_reward, info=None):
        return float(self.theta @ self._phi(s, a))


def make_reward(name, **kwargs):
    registry = {
        "sparse":          SparseReward,
        "pbrs":            PBRSReward,
        "pbrs2":           PBRS2Reward,
        "naive_curiosity": NaiveCuriosityReward,
        "icm":             ICMReward,
        "count_based":     CountBasedReward,
        "irl":             IRLReward,
    }
    if name not in registry:
        raise ValueError(f"Unknown reward: {name}. Choose from {list(registry.keys())}")
    return registry[name](**kwargs)
