"""Class for the UCB policy as described in Chapter 2"""

from policy import Policy
import numpy as np


class UpperConfidenceBound(Policy):
    def __init__(self, c, alpha=0.1):
        super().__init__()
        self.c = c
        self.alpha = alpha
        self.k = self.ten_armed_testbed.get_k()
        self.Q = np.zeros(self.k)
        self.N = np.zeros(self.k)

    def reset(self):
        self.Q = np.zeros(self.k)
        self.N = np.zeros(self.k)
        self.ten_armed_testbed.reset()

    def update_policy(self, action: int, timestep_reward: float):
        self.N[action] += 1

        self.Q[action] = self.Q[action] + (
            self.alpha * (timestep_reward - self.Q[action])
        )

    def choose_action(self) -> int:
        t = max(self.ten_armed_testbed.get_timestep(), 1)
        # If any action has never been taken, choose it (treat as maximizing)
        if np.any(self.N == 0):
            return np.random.choice(np.where(self.N == 0)[0])

        return np.argmax(self.Q + self.c * np.sqrt(np.log(t) / self.N))
