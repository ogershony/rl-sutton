"""Class for the epsilon greedy policy as described in Chapter 2, modified for optimistic q initialization and constant alpha config"""

from policy import Policy
import numpy as np


class EpsilonGreedy(Policy):
    def __init__(self, epsilon, alpha, default_q=0):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.default_q = default_q
        self.k = self.ten_armed_testbed.get_k()
        self.Q = np.array([float(default_q)] * self.k)
        self.N = np.zeros(self.k)

    def reset(self):
        self.Q = np.array([float(self.default_q)] * self.k)
        self.N = np.zeros(self.k)
        self.ten_armed_testbed.reset()

    def update_policy(self, action: int, timestep_reward: float):
        self.N[action] += 1

        if self.alpha is None:
            self.Q[action] = self.Q[action] + (
                (1 / self.N[action]) * (timestep_reward - self.Q[action])
            )
        else:
            self.Q[action] = self.Q[action] + (
                self.alpha * (timestep_reward - self.Q[action])
            )

        # Clips to e^10 for large alpha
        self.Q[action] = np.clip(self.Q[action], -1e10, 1e10)

    def choose_action(self) -> int:
        r = np.random.random()

        # Choose greediliy or randomly depending on epsilon
        if r <= (1 - self.epsilon):
            return np.argmax(self.Q)
        else:
            return np.random.randint(len(self.Q))
