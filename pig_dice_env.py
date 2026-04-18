"""
Pig Dice environment (Gymnasium-compatible).

State: (s_A, s_B, kappa)
    s_A   in {0, ..., 99}   agent's banked score
    s_B   in {0, ..., 105}  opponent's banked score
    kappa in {0, ..., 105}  current turn total
"""

import numpy as np
from collections import namedtuple

ROLL = 0
HOLD = 1
TARGET = 100

_MAX_S_B = 105
_MAX_KAPPA = 105

_ObsSpace = namedtuple("ObsSpace", ["low", "high", "shape", "dtype", "n_states"])


def _compute_n_states():
    return (_MAX_S_B + 1) * sum(_MAX_KAPPA - s_A for s_A in range(TARGET))


class PigDiceEnv:
    """Pig Dice as an MDP with a pluggable opponent policy."""

    observation_space = _ObsSpace(
        low=np.array([0, 0, 0], dtype=np.int32),
        high=np.array([TARGET - 1, _MAX_S_B, _MAX_KAPPA], dtype=np.int32),
        shape=(3,),
        dtype=np.int32,
        n_states=_compute_n_states(),
    )
    action_space = (ROLL, HOLD)

    def __init__(self, opponent_policy=None, agent_starts=True):
        self.opponent_policy = opponent_policy or self._hold_at_20
        self.agent_starts = agent_starts
        self.rng = np.random.default_rng()
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.s_A = 0
        self.s_B = 0
        self.kappa = 0
        self.done = False
        self.winner = None
        if not self.agent_starts:
            self._opponent_turn()
        return self._obs(), {}

    def step(self, action):
        assert not self.done, "Episode is over; call reset()."
        reward = 0.0

        if action == ROLL:
            roll = self.rng.integers(1, 7)
            if roll == 1:
                self.kappa = 0
                self._opponent_turn()
                if self.done:
                    reward = -1.0
            else:
                self.kappa += roll
                if self.s_A + self.kappa >= TARGET:
                    self.done = True
                    self.winner = "agent"
                    reward = 1.0
        else:
            self.s_A += self.kappa
            self.kappa = 0
            if self.s_A >= TARGET:
                self.done = True
                self.winner = "agent"
                reward = 1.0
            else:
                self._opponent_turn()
                if self.done:
                    reward = -1.0

        return self._obs(), reward, self.done, False, {}

    def _obs(self):
        return np.array([self.s_A, self.s_B, self.kappa], dtype=np.int32)

    def state_tuple(self):
        return (self.s_A, self.s_B, self.kappa)

    def _opponent_turn(self):
        opp_kappa = 0
        while True:
            action = self.opponent_policy(self.s_B, self.s_A, opp_kappa)
            if action == ROLL:
                roll = self.rng.integers(1, 7)
                if roll == 1:
                    break
                opp_kappa += roll
                if self.s_B + opp_kappa >= TARGET:
                    self.s_B += opp_kappa
                    self.done = True
                    self.winner = "opponent"
                    return
            else:
                self.s_B += opp_kappa
                if self.s_B >= TARGET:
                    self.done = True
                    self.winner = "opponent"
                    return
                break

    @staticmethod
    def _hold_at_20(my_banked, opp_banked, kappa):
        if my_banked + kappa >= TARGET:
            return HOLD
        return HOLD if kappa >= 20 else ROLL
