"""Class for the epsilon greedy policy as described in Chapter 2, modified for optimistic q initialization and constant alpha config"""

from policy import Policy
import numpy as np


class GradientBandit(Policy):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.k = self.ten_armed_testbed.get_k()
        self.H = np.zeros(self.k)

        # Initialized with equal probability
        self.P = np.full(self.k, 1.0 / self.k)
        self.R = 0  # Track the average reward across timesteps

    def reset(self):
        self.H = np.zeros(self.k)
        self.P = np.full(self.k, 1.0 / self.k)
        self.R = 0
        self.ten_armed_testbed.reset()

    def update_policy(self, action: int, timestep_reward: float):
        # Update the action preference array
        for h in range(len(self.H)):
            # If h is H(a)
            if h == action:
                self.H[h] = self.H[h] + self.alpha * (timestep_reward - self.R) * (
                    1 - self.P[h]
                )

            # h is not H(a)
            else:
                self.H[h] = (
                    self.H[h] - self.alpha * (timestep_reward - self.R) * self.P[h]
                )

        # Update the probability distribution with the new H values
        exp_H = np.exp(self.H - np.max(self.H))
        self.P = exp_H / np.sum(exp_H)

        # Update the average reward
        self.R = self.R + (1 / self.ten_armed_testbed.get_timestep()) * (
            timestep_reward - self.R
        )

    def choose_action(self) -> int:
        # Pick randomly according to probability distribution
        return np.random.choice(len(self.P), p=self.P)
