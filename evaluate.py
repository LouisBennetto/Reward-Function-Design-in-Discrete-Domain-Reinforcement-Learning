import numpy as np
from pig_dice_env import PigDiceEnv, ROLL, HOLD, TARGET


def evaluate_win_rate(agent, opponent_policy, n_episodes=100_000,
                      agent_starts=True, seed=42):
    env = PigDiceEnv(opponent_policy=opponent_policy, agent_starts=agent_starts)
    wins = 0
    total_steps = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep if seed else None)
        done = False
        ep_steps = 0
        while not done:
            state = tuple(obs)
            action = agent.select_action(state, greedy=True)
            obs, _, done, _, _ = env.step(action)
            ep_steps += 1
        if env.winner == "agent":
            wins += 1
        total_steps += ep_steps

    win_rate = wins / n_episodes
    avg_steps = total_steps / n_episodes
    return win_rate, {"wins": wins, "n_episodes": n_episodes, "avg_steps": avg_steps}


def evaluate_policy_win_rate(policy_dict, opponent_policy, n_episodes=100_000,
                             agent_starts=True, seed=42):
    env = PigDiceEnv(opponent_policy=opponent_policy, agent_starts=agent_starts)
    wins = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep if seed else None)
        done = False
        while not done:
            state = tuple(obs)
            action = policy_dict.get(state, ROLL)
            obs, _, done, _, _ = env.step(action)
        if env.winner == "agent":
            wins += 1
    return wins / n_episodes


def compute_policy_deviation(learned_policy, optimal_policy):
    """Fraction of states where learned_policy and optimal_policy disagree."""
    n_disagree = 0

    if isinstance(learned_policy, np.ndarray):
        n_total = len(optimal_policy)
        for (s_A, s_B, kappa), opt_action in optimal_policy.items():
            if int(learned_policy[s_A, s_B, kappa]) != opt_action:
                n_disagree += 1
    else:
        visited = set(learned_policy.keys()) & set(optimal_policy.keys())
        n_total = len(visited)
        for state in visited:
            if learned_policy[state] != optimal_policy[state]:
                n_disagree += 1

    deviation = n_disagree / n_total if n_total > 0 else 0.0
    return deviation, n_disagree, n_total


def compute_reward_density(env_rewards):
    """ρ_R = E[non-zero rewards per episode] / E[T]."""
    total_nonzero = 0
    total_steps = 0
    for episode_rewards in env_rewards:
        total_nonzero += sum(1 for r in episode_rewards if r != 0)
        total_steps += len(episode_rewards)
    if total_steps == 0:
        return 0.0
    return total_nonzero / total_steps
