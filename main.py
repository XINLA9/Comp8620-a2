import argparse

import numpy as np


class Cell:
    def __init__(self, floor_type, cleaning_probability):
        self.floor_type = floor_type
        self.cleaning_probability = cleaning_probability


class GridWorld:
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.grid = [[None for _ in range(n)] for _ in range(m)]

    def set_cell(self, row, col, cell):
        self.grid[row][col] = cell


class Robot:
    def __init__(self, grid_world):
        self.grid_world = grid_world

    def clean(self, row, col):
        cell = self.grid_world.grid[row][col]
        cleaning_probability = cell.cleaning_probability
        return cleaning_probability


def load_scenario_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()
        m, n = map(int, lines[1].split())
        grid_world = GridWorld(m, n)
        row = 0
        for line in lines[4:]:
            cleaning_probabilities = list(map(float, line.split()))
            for col, prob in enumerate(cleaning_probabilities):
                floor_type = "Vinyl"  # 根据实际情况设置地板类型
                cell = Cell(floor_type, prob)
                grid_world.set_cell(row, col, cell)
            row += 1
        return grid_world


def value_iteration(grid_world, discount_factor, theta=0.0001):
    m, n = grid_world.m, grid_world.n
    num_states = m * n
    num_actions = 4  # 上、下、左、右四个动作
    P = np.zeros((num_states, num_actions, num_states))
    R = np.zeros((num_states, num_actions))

    # 创建转移概率矩阵和奖励函数
    for i in range(m):
        for j in range(n):
            state = i * n + j
            for action in range(num_actions):
                if action == 0:  # 上移
                    next_state = max(0, state - n)
                elif action == 1:  # 下移
                    next_state = min(num_states - 1, state + n)
                elif action == 2:  # 左移
                    next_state = max(0, state - 1)
                else:  # 右移
                    next_state = min(num_states - 1, state + 1)

                cell = grid_world.grid[i][j]
                cleaning_probability = cell.cleaning_probability
                P[state, action, next_state] = 1.0
                R[state, action] = -1.0  # 成本为-1

                # 考虑清洁概率影响奖励
                R[state, action] *= (1 - cleaning_probability)

    V = np.zeros(num_states)

    while True:
        delta = 0
        for state in range(num_states):
            v = V[state]
            max_v = float("-inf")
            for action in range(num_actions):
                expected_v = np.sum(P[state, action] * (R[state, action] + discount_factor * V))
                max_v = max(max_v, expected_v)
            V[state] = max_v
            delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break

    policy = np.zeros((num_states, num_actions))
    for state in range(num_states):
        max_v = float("-inf")
        best_action = 0
        for action in range(num_actions):
            expected_v = np.sum(P[state, action] * (R[state, action] + discount_factor * V))
            if expected_v > max_v:
                max_v = expected_v
                best_action = action
        policy[state][best_action] = 1

    return V, policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Value Iteration for Cleaning Robot')
    parser.add_argument('file', type=str, help='Path to the scenario file')
    args = parser.parse_args()

    # 使用命令行参数指定的文件加载场景
    file_path = args.file
    grid_world = load_scenario_from_file(file_path)

    # 执行值迭代
    discount_factor = 0.9
    optimal_values, optimal_policy = value_iteration(grid_world, discount_factor)

    # 返回离线最优策略
    print("Optimal Policy:")
    print(optimal_policy)
