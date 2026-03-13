"""Policy abstract class"""

from abc import ABC, abstractmethod
from ten_armed_testbed import TenArmedTestbed


class Policy(ABC):
    def __init__(self):
        self.ten_armed_testbed = TenArmedTestbed()
        self.timesteps = 200000

    def run_episode(self):
        """Should run an episode with the specific policy"""
        self.reset()

        # Step through the testbed
        for i in range(self.timesteps):
            action = self.choose_action()
            timestep_reward = self.ten_armed_testbed.step(action)
            self.update_policy(action, timestep_reward)

        episode_reward = self.ten_armed_testbed.get_average_reward()

        return episode_reward

    @abstractmethod
    def choose_action(self) -> int:
        """This should determine which action to take, implemented by the policy"""
        pass

    @abstractmethod
    def update_policy(self, action: int, timestep_reward: float):
        """Update the policy based on the action taken and reward received"""
        pass

    @abstractmethod
    def reset(self):
        """Reset policy state and testbed for a new episode"""
        pass
