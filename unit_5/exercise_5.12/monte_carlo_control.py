"""
Implements On-Policy First-Visit Monte Carlo Control (Section 5.4, Sutton & Barto)
to find an optimal policy for the racetrack problem.
"""

import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import random

NUM_EPISODES = 100_000
DISCOUNT = 1.0
EPSILON = 0.1
MAX_STEPS_PER_EPISODE = 1000


class MonteCarloControl:
    def __init__(self, racetrack):
        """Constructor, builds MDP"""
        self.racetrack = racetrack
        self.course = racetrack.get_course()

        self.states = []
        self.actions = []
        self.Q = {}  # Format: {state: {action: q(s,a)}}
        self.N = {}  # Format: {state: {action: visit count}}

        # Build states: (row, col, v_row, v_col)
        for row in range(self.course.shape[0]):
            for col in range(self.course.shape[1]):
                for vel_forward in range(0, 5):
                    for vel_horizontal in range(0, 5):
                        state = (row, col, vel_forward, vel_horizontal)
                        self.states.append(state)

        # Build actions: 9 combinations of (-1, 0, +1) x (-1, 0, +1)
        for action_forward in range(-1, 2):
            for action_horizontal in range(-1, 2):
                action = (action_forward, action_horizontal)
                self.actions.append(action)

        # Initialize Q and N
        for state in self.states:
            self.Q[state] = {}
            self.N[state] = {}
            for action in self.actions:
                self.Q[state][action] = -500
                self.N[state][action] = 0

    def run_monte_carlo_control(self) -> None:
        """On-policy first-visit MC control with ε-greedy policy"""
        for episode_idx in range(NUM_EPISODES):
            episode = self.run_episode()

            if (episode_idx + 1) % 10_000 == 0:
                print(
                    f"Episode {episode_idx + 1}/{NUM_EPISODES}, length: {len(episode)}"
                )

            # First-visit tracking
            visited = set()

            g = 0
            for state, action, reward in reversed(episode):
                g = DISCOUNT * g + reward

                # First-visit check
                if (state, action) in visited:
                    continue
                visited.add((state, action))

                # Incremental mean update
                self.N[state][action] += 1
                self.Q[state][action] += (g - self.Q[state][action]) / self.N[state][
                    action
                ]

    def get_greedy_action(self, state: tuple) -> tuple:
        """Returns the greedy action for a state (argmax Q)"""
        return max(self.Q[state], key=self.Q[state].get)

    def run_episode(self) -> list:
        """Generate an episode using ε-greedy policy"""
        self.racetrack.reset()
        episode = []

        for _ in range(MAX_STEPS_PER_EPISODE):
            state = self.build_state()
            action = self.get_action(state)
            done = self.step(action)
            episode.append((state, action, -1))
            if done:
                break

        return episode

    def build_state(self) -> tuple:
        """Builds the state for the racetrack environment"""
        return self.racetrack.get_car_pos() + self.racetrack.get_velocity()

    def get_action(self, state: tuple, greedy: bool = False) -> tuple:
        """Returns action using ε-greedy policy (or pure greedy if greedy=True)"""
        if not greedy and np.random.random() < EPSILON:
            return random.choice(self.actions)
        return self.get_greedy_action(state)

    def step(self, action) -> bool:
        """Steps through the racetrack environment, returns done if concluded"""
        # Apply action to the racetrack
        self.racetrack.update_velocity(action[0], action[1])

        # Step through the racetrack
        done = self.racetrack.step()

        return done

    def visualize_episode(self, max_steps: int = 200) -> None:
        """Run an episode using the optimal (greedy) policy and save a visualization of the path."""
        self.racetrack.reset()
        positions = [self.racetrack.get_car_pos()]

        for _ in range(max_steps):
            state = self.build_state()
            action = self.get_action(state, greedy=True)
            done = self.step(action)
            positions.append(self.racetrack.get_car_pos())
            if done:
                break

        print(f"Episode length: {len(positions) - 1} steps, finished: {done}")

        # Plot the track
        cmap = mcolors.ListedColormap(["white", "lightgray", "green", "red"])
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(self.course, cmap=cmap, norm=norm, origin="upper")

        print(positions)

        # Overlay the car's path
        rows = [p[0] for p in positions]
        cols = [p[1] for p in positions]
        ax.plot(cols, rows, "b.-", markersize=6, linewidth=1.5)
        ax.plot(cols[0], rows[0], "ko", markersize=10, label="Start")
        ax.plot(cols[-1], rows[-1], "k*", markersize=14, label="End")
        ax.legend(loc="upper right")
        ax.set_title("Optimal Policy Episode")

        diagrams_dir = os.path.join(os.path.dirname(__file__), "episode_diagrams")
        os.makedirs(diagrams_dir, exist_ok=True)
        fig.savefig(
            os.path.join(diagrams_dir, "optimal_episode.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
        print(f"Saved to {diagrams_dir}/optimal_episode.png")

    def sanity_check(self) -> None:
        """Ensure initialization is correct"""

        print(f"States: {len(self.states)}")
        print(f"Actions: {len(self.actions)}")
        print(f"Q: {len(self.Q)} states x {len(self.actions)} actions")
        print(f"N: {len(self.N)} states x {len(self.actions)} actions")
