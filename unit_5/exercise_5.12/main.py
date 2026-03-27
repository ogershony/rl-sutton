"""
Context: Exercise 5.12 (Programming) Consider driving a race car around a turn of a racetrack. You want to go as fast
as possible, but not so fast as to run off the track. In our simplified racetrack, the car is at one of a discrete set
of grid positions, the cells in the diagram. The velocity is also discrete, a number of grid cells moved horizontally
and vertically per time step. The actions are increments to the velocity components. Each may be changed by +1, -1, or 0
in each step, for a total of 9 actions. Both velocity components are restricted to be nonnegative and less than 5, and
they cannot both be zero except at the starting line. Each episode begins in one of the randomly selected start states
with both velocity components zero and ends when the car crosses the finish line. The rewards are -1 for each step until
the car crosses the finish line. If the car hits the track boundary, it is moved back to a random position on the
starting line, both velocity components are reduced to zero, and the episode continues. Apply a Monte Carlo control
method to this task to compute the optimal policy from each starting state.

AI: racetrack.py, tracks.py
Manual: main.py, monte_carlo_control.py
"""

from racetrack import Racetrack
from monte_carlo_control import MonteCarloControl
from tracks import get_track_1, get_track_2


def main():
    track_1 = Racetrack(get_track_1())
    track_2 = Racetrack(get_track_2())

    print(
        f"Track 1: {track_1.course.shape}, start cells: {len(track_1.start_cells)}, finish cells: {len(track_1.finish_cells)}"
    )
    print(
        f"Track 2: {track_2.course.shape}, start cells: {len(track_2.start_cells)}, finish cells: {len(track_2.finish_cells)}"
    )

    track_1_control = MonteCarloControl(track_1)

    track_1_control.sanity_check()

    track_1_control.run_monte_carlo_control()

    track_1_control.visualize_episode()


if __name__ == "__main__":
    main()
