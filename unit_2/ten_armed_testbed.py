"""
Implements the 10 Armed Testbed Class as described in 2.3 & 2.5, with nonstationary tracking.

Nonstationary tracking will be implemented as following: all q* will start at 0, then take a random walk every time step
with a normal distribution with mean 0 and standard deviation 0.01.
"""

import numpy as np


class TenArmedTestbed:
    def __init__(self, k=10):
        self.k = k  # number of arms
        self.timestep = 0

        self.reward = 0  # total reward per time step
        self.reward_threshold = 100000
        self.reward_standard_deviation = 1  # variance to apply to reward

        self.random_walk_mean = 0  # mean of the random walk
        self.random_walk_standard_deviation = (
            0.01  # standard deviation of the random walk
        )

        self.q_star = np.array([0.0] * self.k)  # centered mean per arm

    def reset(self):
        """Resets the episode for the next run"""
        self.reward = 0
        self.timestep = 0
        self.q_star = np.array([0.0] * self.k)

    def random_walk(self):
        self.q_star += np.random.normal(
            self.random_walk_mean,
            self.random_walk_standard_deviation,
            size=self.q_star.shape,
        )

    def step(self, action: int) -> int:
        """Steps the environment forward by incrementing reward, returns the reward"""

        timestep_reward = np.random.normal(
            self.q_star[action], self.reward_standard_deviation
        )

        # Only count reward past reward theshold
        if self.timestep >= self.reward_threshold:
            self.reward += timestep_reward

        self.random_walk()

        self.timestep += 1

        return timestep_reward

    def get_average_reward(self) -> int:
        """Gets average reward per timestep past the reward threshold"""

        if self.timestep < self.reward_threshold:
            return 0.0
        else:
            return self.reward / (self.timestep - self.reward_threshold)

    def get_qstar(self) -> np.array:
        return self.q_star

    def get_k(self) -> int:
        return self.k

    def get_timestep(self) -> int:
        return self.timestep
