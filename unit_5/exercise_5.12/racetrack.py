"""
Holds the Racetrack class, which represents the racetrack environment for Exercise 5.12.
The car starts on the starting line and must reach the finish line by adjusting its velocity.
"""

import numpy as np


# Cell types
OFF_TRACK = 0
TRACK = 1
START = 2
FINISH = 3


class Racetrack:
    def __init__(self, course: np.ndarray):
        self.course = course
        self.rows, self.cols = course.shape

        self.start_cells = list(zip(*np.where(course == START)))
        self.finish_cells = set(zip(*np.where(course == FINISH)))

        self.row = 0
        self.col = 0
        self.v_row = 0
        self.v_col = 0
        self._done = False

        self.reset()

    def reset(self) -> tuple:
        """Place car at a random start-line cell with zero velocity. Returns initial position."""
        idx = np.random.randint(len(self.start_cells))
        self.row, self.col = self.start_cells[idx]
        self.v_row = 0
        self.v_col = 0
        self._done = False
        return (self.row, self.col)

    def update_velocity(self, accel_row: int, accel_col: int):
        """
        Update velocity by acceleration components in {-1, 0, +1}.
        Each velocity component is clamped to [0, 4].
        If the update would make both components zero, it is not applied.
        """
        new_v_row = np.clip(self.v_row + accel_row, 0, 4)
        new_v_col = np.clip(self.v_col + accel_col, 0, 4)

        if new_v_row == 0 and new_v_col == 0:
            return

        self.v_row = int(new_v_row)
        self.v_col = int(new_v_col)

    def step(self) -> bool:
        """
        Move the car by its current velocity.
        If the car reaches a finish cell, sets done to True.
        If the car goes off-track or out of bounds, resets to a random start position.
        Returns done.
        """
        new_row = self.row - self.v_row  # negative row = upward on the grid
        new_col = self.col + self.v_col

        # Check if path crosses finish line (simple check: does the destination land on finish)
        if self._crosses_finish(self.row, self.col, new_row, new_col):
            self._done = True
            return True

        # Check if out of bounds or off track
        if (
            new_row < 0
            or new_row >= self.rows
            or new_col < 0
            or new_col >= self.cols
            or self.course[new_row, new_col] == OFF_TRACK
        ):
            self.reset()
            return False

        self.row = new_row
        self.col = new_col
        return False

    def _crosses_finish(self, r0: int, c0: int, r1: int, c1: int) -> bool:
        """Check if the path from (r0, c0) to (r1, c1) crosses any finish cell."""
        # Use Bresenham-style enumeration of cells along the path
        dr = r1 - r0
        dc = c1 - c0
        steps = max(abs(dr), abs(dc), 1)

        for i in range(1, steps + 1):
            r = r0 + round(dr * i / steps)
            c = c0 + round(dc * i / steps)
            if (r, c) in self.finish_cells:
                return True
        return False

    def get_car_pos(self) -> tuple:
        return (int(self.row), int(self.col))

    def get_velocity(self) -> tuple:
        return (int(self.v_row), int(self.v_col))

    def get_course(self) -> np.ndarray:
        """Returns the racetrack course"""
        return self.course

    def done(self) -> bool:
        """Returns whether the car has reached the finish line."""
        return self._done
