"""
This module implements exercise 2.11 from the book "Reinforcement Learning: An Introduction" by Sutton and Barto.

Reference:
Exercise 2.11 (Page 44): Make a figure analogous to figure 2.6 for the nonstationary case outlined in Exercise 2.5.
Include the constant-step-size e-greedy algorithm with alpha=0.1. Use runs of 200,000 steps and, as a performance measure
for each algorithm and parameter settings, use the average reward over the last 100,000 steps.
"""

import matplotlib.pyplot as plt
import numpy as np

from epsilon_greedy import EpsilonGreedy
from upper_confidence_bound import UpperConfidenceBound
from gradient_bandit import GradientBandit

episodes = 10  # Number of episodes to average over

# Param range to test (e, a, c, q)
param_range = [
    float(1 / 128),
    float(1 / 64),
    float(1 / 32),
    float(1 / 16),
    float(1 / 8),
    float(1 / 2),
    1,
    2,
    4,
]

# Rewards for each policy
(
    eg_rewards,
    eg_constant_rewards,
    eg_optimistic_rewards,
    ucb_rewards,
    gradient_rewards,
) = [], [], [], [], []


def main():
    # Log rewards for each parameter for each policy
    for param in param_range:
        print(f"Logging rewards for param value {param}")

        eg_rewards.append(get_average_episode_reward(EpsilonGreedy(param, None)))
        eg_constant_rewards.append(
            get_average_episode_reward(EpsilonGreedy(0.1, param))
        )
        eg_optimistic_rewards.append(
            get_average_episode_reward(EpsilonGreedy(0.1, 0.1, default_q=param))
        )
        ucb_rewards.append(get_average_episode_reward(UpperConfidenceBound(param)))
        gradient_rewards.append(get_average_episode_reward(GradientBandit(param)))

    build_graph()


def get_average_episode_reward(policy):
    # Gets the episode reward averaged over num episodes
    total_reward = 0
    for i in range(episodes):
        episode_reward = policy.run_episode()
        total_reward += episode_reward

    return total_reward / episodes


def build_graph():
    x = np.arange(len(param_range))
    x_labels = []
    for p in param_range:
        if p >= 1:
            x_labels.append(str(int(p)))
        else:
            denom = int(round(1 / p))
            x_labels.append(f"1/{denom}")

    plt.figure()
    plt.plot(x, eg_rewards, label="ε-greedy (varying ε)")
    plt.plot(x, eg_constant_rewards, label="ε-greedy (constant α, ε=0.1)")
    plt.plot(x, eg_optimistic_rewards, label="ε-greedy (optimistic, ε=0.1)")
    plt.plot(x, ucb_rewards, label="UCB (varying c)")
    plt.plot(x, gradient_rewards, label="Gradient Bandit (varying α)")
    plt.xticks(x, x_labels)
    plt.xlabel("Parameter Value (log2 scale)")
    plt.ylabel("Average Reward")
    plt.title("Parameter Study (Exercise 2.11)")
    plt.legend()
    plt.savefig("policy_performance.png")
    plt.show()


if __name__ == "__main__":
    main()
