"""
Holds the MDP class, which represents Jacks' Car Rental Problem as a finite MDP
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson


class MDP:
    def __init__(
        self,
        max_cars_per_lot: int,
        max_action: int,
        moving_car_reward: int,
        renting_car_reward: int,
        mean_requests_1: int,
        mean_requests_2: int,
        mean_returns_1: int,
        mean_returns_2: int,
        requests_car_limit=7,
        returns_car_limit=7,
        discount_factor=0.9,
    ):
        self.requests_car_limit = (
            requests_car_limit  # Need a ceiling on Poisson distribution
        )
        self.returns_car_limit = (
            returns_car_limit  # Need a ceiling on Poisson distribution
        )
        self.discount_factor = discount_factor
        self.max_cars_per_lot = max_cars_per_lot
        self.max_action = max_action
        self.moving_car_reward = moving_car_reward
        self.renting_car_reward = renting_car_reward
        self.mean_requests_1 = mean_requests_1
        self.mean_requests_2 = mean_requests_2
        self.mean_returns_1 = mean_returns_1
        self.mean_returns_2 = mean_returns_2

        # Precompute Poisson probabilities lookup table
        self.poisson_table = {}
        for lam, limit in [
            (mean_returns_1, returns_car_limit),
            (mean_requests_1, requests_car_limit),
            (mean_returns_2, returns_car_limit),
            (mean_requests_2, requests_car_limit),
        ]:
            for k in range(limit):
                self.poisson_table[(k, lam)] = poisson.pmf(k, lam)

        self.setup_MDP()

    def setup_MDP(self):
        """Initializes states, state values, actions, and a policy, set to an even probability distribution"""
        self.states = []  # Holds state ids
        self.state_values = {}  # Value function at each state (key: state id, value: v(s))

        self.actions = []  # Holds action ids
        self.policy = {}  # Policy at a given state (key: state id, value: {key: action id, value: p(a | s)})

        # Initialize states
        for i in range(self.max_cars_per_lot + 1):
            for j in range(self.max_cars_per_lot + 1):
                state_id = (i, j)
                self.states.append(state_id)
                # Initialize with 0 value
                self.state_values[state_id] = 0.0

        # Initialize actions
        for i in range((-1) * self.max_action, self.max_action + 1):
            action_id = i
            self.actions.append(action_id)

        # Initialize policy
        for state_id in self.states:
            self.policy[state_id] = {}
            # Only include valid actions
            valid_actions = []
            for action_id in self.actions:
                # Checks if the actions are valid
                if action_id >= 0:
                    if state_id[0] >= action_id:
                        valid_actions.append(action_id)
                else:
                    if state_id[1] >= action_id * -1:
                        valid_actions.append(action_id)

            for action_id in valid_actions:
                # Initialize with even probability distribution
                self.policy[state_id][action_id] = 1.0 / len(valid_actions)

    def run_policy_iteration(self, theta=0.01) -> dict:
        """Performs the policy iteration algorithm to obtain optimal policy and value functions."""

        # Clear old policy diagrams
        diagrams_dir = os.path.join(os.path.dirname(__file__), "policy_diagrams")
        os.makedirs(diagrams_dir, exist_ok=True)
        for f in os.listdir(diagrams_dir):
            if f.endswith(".png"):
                os.remove(os.path.join(diagrams_dir, f))

        policy_stable = False
        epoch = 0

        self.visualize_policy("epoch_0")

        # Iterate until the policy stabilizes
        while policy_stable is not True:
            print(f"Running policy iteration on epoch {epoch}")
            self.evaluate_policy(theta)
            policy_stable = self.improve_policy()

            epoch += 1
            self.visualize_policy(f"epoch_{epoch}")

        return self.policy

    def visualize_policy(self, epoch_name: str):
        """Saves a heatmap of the current policy to policy_diagrams/{epoch_name}.png"""
        n = self.max_cars_per_lot + 1
        policy_grid = np.zeros((n, n), dtype=int)

        for (i, j), actions in self.policy.items():
            # Extract the deterministic action (highest probability)
            optimal_action = max(actions, key=actions.get)
            policy_grid[j, i] = optimal_action  # y=lot2, x=lot1

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.pcolormesh(
            np.arange(n + 1) - 0.5,
            np.arange(n + 1) - 0.5,
            policy_grid,
            cmap="RdBu",
            shading="flat",
        )
        plt.colorbar(im, ax=ax, label="Cars moved (lot1 → lot2)")
        ax.set_xlabel("Cars at lot 1")
        ax.set_ylabel("Cars at lot 2")
        ax.set_title(f"Policy — {epoch_name}")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_aspect("equal")

        diagrams_dir = os.path.join(os.path.dirname(__file__), "policy_diagrams")
        fig.savefig(
            os.path.join(diagrams_dir, f"{epoch_name}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    def evaluate_policy(self, theta: float):
        """
        Uses dynamic programming to update the state values, which due to the Bellman
        optimality equation is guaranteed to converge on v*
        """

        delta = float("inf")
        epoch = 0

        # Until the biggest change is less than some small value (state values have converged to v*)
        while delta > theta:
            delta = 0.0
            for state in self.states:
                next_state_value = 0.0
                actions = self.policy[state]
                for action, p_action in actions.items():
                    # Get q(s,a)
                    state_action_value = self.get_state_action_value(state, action)

                    # Now multiply it by pi(s | a) and sum to get v(s)
                    next_state_value += state_action_value * p_action

                # Track the biggest change in state values
                delta = max(delta, abs(self.state_values[state] - next_state_value))

                # Update the state value
                self.state_values[state] = next_state_value

            print(f"Policy evaluation epoch {epoch}, delta = {delta}")
            epoch += 1

    def improve_policy(self) -> bool:
        """Updates the existing policy to the new optimal action, which is guaranteed to converge to pi*"""

        policy_stable = True

        print("Policy evaluation complete, improving policy...")
        for state in self.states:
            actions = self.policy[state]
            # Finds the action in state s with the highest policy probability
            optimal_action = max(actions, key=actions.get)

            # Finds the new optimal action based on the highest state value of taking each action
            new_optimal_action = max(
                actions, key=lambda a: self.get_state_action_value(state, a)
            )

            # Now update policy deterministically with new optimal action
            self.policy[state] = {
                k: (1.0 if k == new_optimal_action else 0.0)
                for k, v in self.policy[state].items()
            }

            # If optimal action changed, then policy is not stable
            if optimal_action != new_optimal_action:
                policy_stable = False

        if not policy_stable:
            print("Policy not stable, new action detected...")

        else:
            print("Policy stable, exiting...")

        return policy_stable

    def get_state_action_value(self, state, action):
        """Effectively calculates q(s,a): returns value of taking a particular action at a particular state"""

        total_value = 0.0

        # Immediate reward for taking action
        reward = abs(action) * self.moving_car_reward

        # First applies the action to the state
        delta_state = (state[0] - action, state[1] + action)

        # Goes through all possible state changes
        for i in range(self.returns_car_limit):
            for j in range(self.requests_car_limit):
                for k in range(self.returns_car_limit):
                    for l in range(self.requests_car_limit):
                        next_state = (delta_state[0] + i - j, delta_state[1] + k - l)

                        # If next state is a valid state
                        if next_state in self.states:
                            # Compute probability of arriving to s'
                            next_state_prob = (
                                self.get_poisson_prob(i, self.mean_returns_1)
                                * self.get_poisson_prob(j, self.mean_requests_1)
                                * self.get_poisson_prob(k, self.mean_returns_2)
                                * self.get_poisson_prob(l, self.mean_requests_2)
                            )

                            # Value of arriving to s' (r + discount * V'(s'))
                            next_state_reward = reward + self.renting_car_reward * (
                                j + l
                            )
                            next_state_value = (
                                next_state_reward
                                + self.discount_factor * self.state_values[next_state]
                            )

                            # Now multiply the prob of arrriving to s' by its value of getting there, and add to total value
                            total_value += next_state_prob * next_state_value

        return total_value

    def get_poisson_prob(self, k, lam):
        """Returns the probability of an integer occuring in a Poisson random distribution with mean lambda"""
        return self.poisson_table[(k, lam)]

    def get_policy(self):
        """Sanity check for policy initialization"""
        return self.policy
