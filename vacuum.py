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
                    state_str = 'x' + str(i) + 'y' + str(j) + ''.join(state)
                    self._states.append(state_str)
        print(self._states)

        self._actions = ['up', 'down', 'left', 'right', 'vacuum']
        self.gamma = 0.99
        self.theta = 1e-3
        self._transitions = {}
        self._rewards = {}
        self.V = {}
        for state in self._states:
            self.V[state] = 0

    def transition(self, state, action):
        match = re.match(r'x(\d+)y(\d+)(.*)', state)
        if not match:
            return {state: 1.0}  # 返回原始状态的字典，如果不匹配

        i, j, cleanliness = int(match.group(1)), int(match.group(2)), match.group(3)

        if action == 'up':
            i = max(0, i - 1)
        elif action == 'down':
            i = min(self._m - 1, i + 1)
        elif action == 'left':
            j = max(0, j - 1)
        elif action == 'right':
            j = min(self._n - 1, j + 1)

        result_states = {'x' + str(i) + 'y' + str(j) + cleanliness: 1.0}

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
            cleaned_state = 'x' + str(i) + 'y' + str(j) + ''.join(cleanliness_list)

            result_states = {
                cleaned_state: cleaning_success_probability,
                'x' + str(i) + 'y' + str(j) + cleanliness: 1 - cleaning_success_probability
            }

        return result_states

    def reward(self, state, action):
        i = int(state[1])
        j = int(state[3])
        cell_type = self._grids[i][j]
        if action == 'vacuum':
            if cell_type == 'v':
                return 10
            elif cell_type == 't':
                return 8
            elif cell_type == 'T':
                return 5
        return -1

    def value_iteration(self):
        while True:
            delta = 0
            V_copy = self.V.copy()
            for state in self._states:
                v = self.V[state]
                vals = []
                for action in self._actions:
                    new_state = self.transition(state, action)
                    vals.append(self.reward(state, action) + self.gamma * self.V[new_state])
                V_copy[state] = max(vals)
                delta = max(delta, abs(v - V_copy[state]))
            self.V = V_copy
            if delta < self.theta:
                break
        return self.V


def read_grid_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        return [list(line.strip()) for line in lines]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        filename = "test_case.txt"
    else:
        filename = sys.argv[1]
    grids = read_grid_from_file(filename)
    print(grids)
    mdp = MDP(grids)
    policy = mdp.value_iteration()
    print(policy)
