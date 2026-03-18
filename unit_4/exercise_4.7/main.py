"""
Context: Exercise 4.7 (Programming) Write a program for policy iteration and re-solve Jack's car rental problem with the following changes.
One of Jack's employees at the first location rides a bus home each night and lives near the second location. She is happy to shuttle
one car to the second location for free. Each additional car still costs $2, as do all cars moved in the other direction. In additon,
Jack has limited parking space at each location. If more than 10 cars are kept overnight at a location (after any moving of cars), then
an additional cost of $4 must be incurred to use a second parking lot. To check your program, first replicate the results given for the
original problem.
"""

from mdp import MDP


def main():
    mdp = MDP(20, 5, -2, 10, 3, 4, 3, 2)
    print(f" Original policy: {mdp.get_policy()}")
    optimal_policy = mdp.run_policy_iteration()
    print(f" Optimal policy: {optimal_policy}")


if __name__ == "__main__":
    main()
