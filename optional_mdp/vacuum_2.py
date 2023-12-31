from itertools import product
import re
import sys
import numpy as np


class MDP:
    def __init__(self, _grids):
        self._grids = _grids
        self._m = len(_grids)
        self._n = len(_grids[0])

        self._states = list(product(range(self._m), range(self._n), product('cd', repeat=self._m * self._n)))
        self._goalStates = list(product(range(self._m), range(self._n), product('c', repeat=self._m * self._n)))

        self._actions = ['up', 'down', 'left', 'right']
        self.gamma = 0.90
        self.theta = 1e-3
        self.V = {}
        for state in self._states:
            if state in self._goalStates:
                self.V[state] = 100
            else:
                self.V[state] = 0

    def transition(self, state, action):
        m, n, cleanliness = state
        original_m, original_n = m, n
        if action == 'up':
            m = max(0, m - 1)
        elif action == 'down':
            m = min(self._m - 1, m + 1)
        elif action == 'left':
            n = max(0, n - 1)
        elif action == 'right':
            n = min(self._n - 1, n + 1)

        # Check whether the position of the robot is changed
        if m == original_m and n == original_n:
            return {state: 1.0}  # if the robot stay in the original position, return

        cell_type = self._grids[m][n]
        cleaning_success_probability = 0
        if cell_type == 'v':
            cleaning_success_probability = 0.95
        elif cell_type == 't':
            cleaning_success_probability = 0.85
        elif cell_type == 'T':
            cleaning_success_probability = 0.75

        cleanliness_list = list(cleanliness)
        cleanliness_list[m * self._n + n] = 'c'

        result_states = {
            (m, n, tuple(cleanliness_list)): cleaning_success_probability,
            (m, n, cleanliness): 1 - cleaning_success_probability
        }

        return result_states

    def reward(self, state, action, next_state):
        i, j, cleanliness = state
        i_next, j_next, cleanliness_next = next_state

        if state == next_state:
            return -5
        elif cleanliness_next[i_next * self._n + j_next] == 'c' and cleanliness[i * self._n + j] != 'c':
            return 5
        elif cleanliness_next[i_next * self._n + j_next] == 'c' and cleanliness[i * self._n + j] == 'c':
            return -1
        elif cleanliness_next[i_next * self._n + j_next] != 'c':
            return -3
        else:
            return 0

    def value_iteration(self):
        policy = {}
        iteration_number = 0
        while True:
            delta = 0
            V_copy = self.V.copy()
            for state in self._states:
                if state not in self._goalStates:
                    v = self.V[state]
                    best_action_val = float('-inf')
                    best_action = None
                    for action in self._actions:
                        val = 0
                        transition_probs = self.transition(state, action)
                        for next_state, prob in transition_probs.items():
                            reward_val = self.reward(state, action, next_state)
                            val += prob * (reward_val + self.gamma * self.V[next_state])
                        if val > best_action_val:
                            best_action_val = val
                            best_action = action
                    V_copy[state] = best_action_val
                    policy[state] = best_action
                    delta = max(delta, abs(v - V_copy[state]))
            self.V = V_copy
            iteration_number += 1
            if delta < self.theta:
                break
        return self.V, policy, iteration_number


def read_grid_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        return [list(line.strip()) for line in lines]


def save_solution_to_file(filename, v, policy):
    with open(filename, 'w') as f:
        f.write("value\n")
        for key in v:
            f.write(f"{key} {v[key]}\n")
        f.write("policy\n")
        for key in policy:
            f.write(f"{key} {policy[key]}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        input_filename = "../test/test_case_1"
    else:
        input_filename = sys.argv[1]

    # Extract the test case number and create the output filename
    test_case_num = input_filename.split("_")[-1]  # Get the number after "test_case_"
    output_filename = f"test/solution_{test_case_num}"

    grids = read_grid_from_file(input_filename)
    for row in grids:
        print(row)
    mdp = MDP(grids)
    v, policy, iteration = mdp.value_iteration()

    print(len(policy))
    save_solution_to_file(output_filename, v, policy)