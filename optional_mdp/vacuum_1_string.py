import re
from itertools import product
from typing import List
import numpy as np
import sys


class MDP:
    def __init__(self, _grids):
        self._grids = _grids
        self._m = len(_grids)
        self._n = len(_grids[0])

        self._states = []
        for i in range(self._m):
            for j in range(self._n):
                for state in product('cd', repeat=self._m * self._n):
                    state_str = 'm' + str(i) + 'n' + str(j) + ''.join(state)
                    self._states.append(state_str)
        self._goalStates = []
        for i in range(self._m):
            for j in range(self._n):
                for state in product('c', repeat=self._m * self._n):
                    state_str = 'm' + str(i) + 'n' + str(j) + ''.join(state)
                    self._goalStates.append(state_str)

        self._actions = ['up', 'down', 'left', 'right', 'vacuum']
        self.gamma = 0.90
        self.theta = 1e-3
        self.V = {}
        for state in self._states:
            if state in self._goalStates:
                self.V[state] = 100
            else:
                self.V[state] = 0

    def transition(self, state, action):
        match = re.match(r'm(\d+)n(\d+)(.*)', state)
        if not match:
            return {state: 1.0}

        i, j, cleanliness = int(match.group(1)), int(match.group(2)), match.group(3)

        if action == 'up':
            i = max(0, i - 1)
        elif action == 'down':
            i = min(self._m - 1, i + 1)
        elif action == 'left':
            j = max(0, j - 1)
        elif action == 'right':
            j = min(self._n - 1, j + 1)

        result_states = {'m' + str(i) + 'n' + str(j) + cleanliness: 1.0}

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
            cleaned_state = 'm' + str(i) + 'n' + str(j) + ''.join(cleanliness_list)

            result_states = {
                cleaned_state: cleaning_success_probability,
                'm' + str(i) + 'n' + str(j) + cleanliness: 1 - cleaning_success_probability
            }

        return result_states

    def reward(self, state, action, next_state):
        match_state = re.match(r'm(\d+)n(\d+)(.*)', state)
        match_next_state = re.match(r'm(\d+)n(\d+)(.*)', next_state)

        if not match_state or not match_next_state:
            return 0

        i, j, cleanliness = int(match_state.group(1)), int(match_state.group(2)), match_state.group(3)
        i_next, j_next, cleanliness_next = int(match_next_state.group(1)), int(
            match_next_state.group(2)), match_next_state.group(3)

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
