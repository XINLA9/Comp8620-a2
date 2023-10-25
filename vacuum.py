import sys
from itertools import product
import numpy as np


class MDP:
    def __init__(self, _grids):
        """

        :param _grids:
        """
        self._grids = _grids
        self._m = len(_grids)
        self._n = len(_grids[0])

        self._states = list(product(range(self._m), range(self._n), product('cd', repeat=self._m * self._n)))
        self._goalStates = list(product(range(self._m), range(self._n), product('c', repeat=self._m * self._n)))

        self._actions = ['up', 'down', 'left', 'right', 'vacuum']
        self.gamma = 0.90
        self.theta = 1e-3
        self.V = {}
        for state in self._states:
            if state[:2] + (tuple(state[2]),) in self._goalStates:
                self.V[state] = 100
            else:
                self.V[state] = 0

    @property
    def state_num(self):
        return len(self._states)

    def transition(self, state, action):
        i, j, cleanliness = state
        if action == 'up':
            i = max(0, i - 1)
        elif action == 'down':
            i = min(self._m - 1, i + 1)
        elif action == 'left':
            j = max(0, j - 1)
        elif action == 'right':
            j = min(self._n - 1, j + 1)

        result_states = {(i, j, cleanliness): 1.0}

        if action == 'vacuum':
            cell_type = self._grids[i][j]
            cleaning_success_probability = 0
            if cell_type == 'v':
                cleaning_success_probability = 0.95
            elif cell_type == 't':
                cleaning_success_probability = 0.85
            elif cell_type == 'T':
                cleaning_success_probability = 0.75

            cleanliness_list = list(cleanliness)
            cleanliness_list[i * self._n + j] = 'c'

            result_states = {
                (i, j, tuple(cleanliness_list)): cleaning_success_probability,
                (i, j, cleanliness): 1 - cleaning_success_probability
            }

        return result_states

    def reward(self, state, action, next_state):
        """

        :param state:
        :param action:
        :param next_state:
        :return:
        """
        i, j, cleanliness = state
        i_next, j_next, cleanliness_next = next_state

        if action == 'vacuum':
            if cleanliness[i * self._n + j] == 'c':
                return -5
            elif cleanliness[i * self._n + j] == 'd' and cleanliness_next[i_next * self._n + j_next] == 'c':
                return 10
            elif cleanliness[i * self._n + j] == 'd' and cleanliness_next[i_next * self._n + j_next] == 'd':
                return -1
        elif action in ['up', 'down', 'left', 'right']:
            if state == next_state:
                return -5
            else:
                return -1

        return 0

    def value_iteration(self):
        """

        :return:
        """
        iteration_number = 0
        policy = {}
        while True:
            delta = 0
            V_copy = self.V.copy()
            for state in self._states:
                if state[:2] + (tuple(state[2]),) not in self._goalStates:
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
    """
    Read grid configuration from a file.

    :param filename: String, name of the file.
    :return: List of Lists, the grid configuration.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
        return [list(line.strip()) for line in lines]


def save_solution_to_file(filename, v, policy):
    """
    Save the value function and policy to a file.

    :param filename: String, name of the file.
    :param v: Dictionary with states as keys and their values.
    :param policy: Dictionary with states as keys and their optimal actions.
    :return: None
    """
    with open(filename, 'w') as f:
        f.write("value\n")
        for key in v:
            f.write(f"{key} {v[key]}\n")
        f.write("policy\n")
        for key in policy:
            f.write(f"{key} {policy[key]}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        input_filename = "test/test_case_1"
    else:
        input_filename = "test/" + sys.argv[1]

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