import numpy as np

# Parameters
m, n = 1, 4
gamma = 0.99
theta = 1e-3

# States
states = [(i, j, k, l, m, n) for i in range(m) for j in range(n) for k in [0, 1] for l in [0, 1] for m in [0, 1] for n
          in [0, 1]]

actions = ['up', 'down', 'left', 'right', 'vacuum']


def transition(state, action):
    x, y, c1, c2, c3, c4 = state

    if action == 'up':
        x = max(0, x - 1)
    elif action == 'down':
        x = min(m - 1, x + 1)
    elif action == 'left':
        y = max(0, y - 1)
    elif action == 'right':
        y = min(n - 1, y + 1)
    elif action == 'vacuum':
        if (x, y) == (0, 0):
            c1 = 1
        elif (x, y) == (0, 1):
            c2 = 1
        elif (x, y) == (0, 2):
            c3 = 1
        elif (x, y) == (0, 3):
            c4 = 1

    return x, y, c1, c2, c3, c4


def reward(state, action):
    x, y, c1, c2, c3, c4 = state
    if action == 'vacuum':
        if (x, y) == (0, 0) and c1 == 0:
            return 10
        elif (x, y) == (0, 1) and c2 == 0:
            return 10
        elif (x, y) == (0, 2) and c3 == 0:
            return 10
        elif (x, y) == (0, 3) and c4 == 0:
            return 10
        else:
            return -5
    else:
        return -1


V = {}
for state in states:
    V[state] = 0

while True:
    delta = 0
    V_copy = V.copy()
    for state in states:
        v = V[state]
        vals = []
        for action in actions:
            new_state = transition(state, action)
            vals.append(reward(state, action) + gamma * V[new_state])
        V_copy[state] = max(vals)
        delta = max(delta, abs(v - V_copy[state]))
    V = V_copy
    if delta < theta:
        break

print(V)