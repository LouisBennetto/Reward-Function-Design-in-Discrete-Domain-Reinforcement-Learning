import random as _random
import numpy as np

ROLL = 0
HOLD = 1

_MAX_SA = 100
_MAX_SB = 106
_MAX_K = 106


class MCControlAgent:
    """Every-visit MC Control with epsilon-greedy exploration and optimistic initialisation."""

    def __init__(self, n_actions=2, epsilon_0=1.0, decay_rate=0.99999,
                 optimistic_init=0.5, gamma=1.0):
        self.n_actions = n_actions
        self.epsilon = float(epsilon_0)
        self.epsilon_0 = float(epsilon_0)
        self.decay_rate = float(decay_rate)
        self.gamma = float(gamma)

        self.Q = np.full((_MAX_SA, _MAX_SB, _MAX_K, n_actions),
                         float(optimistic_init), dtype=np.float64)
        self.N = np.zeros((_MAX_SA, _MAX_SB, _MAX_K, n_actions), dtype=np.int32)

    def select_action(self, state, greedy=False):
        s_A, s_B, kappa = state
        if not greedy and _random.random() < self.epsilon:
            return _random.randrange(self.n_actions)
        q0 = self.Q[s_A, s_B, kappa, 0]
        q1 = self.Q[s_A, s_B, kappa, 1]
        if q0 > q1:
            return 0
        if q1 > q0:
            return 1
        return _random.randrange(self.n_actions)

    def update_episode(self, trajectory):
        G = 0.0
        for t in range(len(trajectory) - 1, -1, -1):
            (s_A, s_B, kappa), action, reward = trajectory[t]
            G = reward + self.gamma * G
            self.N[s_A, s_B, kappa, action] += 1
            n = self.N[s_A, s_B, kappa, action]
            self.Q[s_A, s_B, kappa, action] += (
                (G - self.Q[s_A, s_B, kappa, action]) / n
            )

    def decay_epsilon(self):
        self.epsilon *= self.decay_rate

    def get_policy(self):
        return np.argmax(self.Q, axis=3).astype(np.int8)

    def get_q_table(self):
        return self.Q.copy()


class QLearningAgent:
    """Off-policy TD control with epsilon-greedy exploration and polynomial learning-rate decay."""

    def __init__(self, n_actions=2, epsilon_0=1.0, decay_rate=0.99999,
                 alpha_0=0.5, decay_control=1000.0, optimistic_init=0.5,
                 gamma=1.0):
        self.n_actions = n_actions
        self.epsilon = float(epsilon_0)
        self.epsilon_0 = float(epsilon_0)
        self.decay_rate = float(decay_rate)
        self.alpha = float(alpha_0)
        self.alpha_0 = float(alpha_0)
        self.decay_control = float(decay_control)
        self.gamma = float(gamma)
        self.update_count = 1

        self.Q = np.full((_MAX_SA, _MAX_SB, _MAX_K, n_actions),
                         float(optimistic_init), dtype=np.float64)

    def select_action(self, state, greedy=False):
        s_A, s_B, kappa = state
        if not greedy and _random.random() < self.epsilon:
            return _random.randrange(self.n_actions)
        q0 = self.Q[s_A, s_B, kappa, 0]
        q1 = self.Q[s_A, s_B, kappa, 1]
        if q0 > q1:
            return 0
        if q1 > q0:
            return 1
        return _random.randrange(self.n_actions)

    def update_step(self, state, action, reward, next_state, done):
        s_A, s_B, kappa = state
        if done:
            target = reward
        else:
            ns_A, ns_B, nk = next_state
            target = reward + self.gamma * max(
                self.Q[ns_A, ns_B, nk, 0], self.Q[ns_A, ns_B, nk, 1]
            )

        self.Q[s_A, s_B, kappa, action] += self.alpha * (
            target - self.Q[s_A, s_B, kappa, action]
        )

        # α_i = α₀ · c / (c + i)  —  polynomial decay
        c = self.decay_control
        i = self.update_count
        self.alpha *= (c + i - 1.0) / (c + i)
        self.update_count += 1

    def decay_epsilon(self):
        self.epsilon *= self.decay_rate

    def get_policy(self):
        return np.argmax(self.Q, axis=3).astype(np.int8)

    def get_q_table(self):
        return self.Q.copy()
