import time
import matplotlib.pyplot as plt
import numpy as np

from vacuum_1 import MDP


def analyze_scalability(min_size, max_size):
    times = []
    iterations = []

    for size in range(min_size, max_size + 1):
        # Generate a grid of given size with random values ('v', 't', 'T')
        grid = [[np.random.choice(['v', 't', 'T']) for _ in range(size)] for _ in range(size)]

        mdp = MDP(grid)

        start_time = time.time()
        v, policy = mdp.value_iteration()
        end_time = time.time()

        elapsed_time = end_time - start_time
        times.append(elapsed_time)

        # For simplicity, we use the size of V as the number of iterations
        iterations.append(len(v))

    return times, iterations


# Example usage
times, iterations = analyze_scalability(2, 10)

# Plotting results (assuming you have matplotlib installed)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(2, 11), times, 'o-')
plt.xlabel('Grid Size')
plt.ylabel('Time (s)')
plt.title('Time vs Grid Size')

plt.subplot(1, 2, 2)
plt.plot(range(2, 11), iterations, 'o-')
plt.xlabel('Grid Size')
plt.ylabel('Number of Iterations')
plt.title('Iterations vs Grid Size')

plt.tight_layout()
plt.show()
