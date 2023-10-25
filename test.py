import os
import time

from vacuum import MDP


def read_grid_from_file(filename):
    """

    :param filename:
    :return:
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
        return [list(line.strip()) for line in lines]


def save_solution_to_file(filename, v, policy):
    """

    :param filename:
    :param v:
    :param policy:
    :return:
    """
    with open(filename, 'w') as f:
        for key in v:
            f.write(f"{key} {v[key]}\n")
        for key in policy:
            f.write(f"{key} {policy[key]}\n")


def run_value_iteration_for_file(filepath:str):
    """

    :param filepath:
    :return:
    """
    grids = read_grid_from_file(filepath)
    mdp = MDP(grids)
    state_num = mdp.state_num
    m = len(grids)
    n = len(grids[0])

    start_time = time.time()
    v, policy, num_iterations = mdp.value_iteration()
    end_time = time.time()

    elapsed_time = end_time - start_time
    num_policies = len(policy)

    return m, n, state_num, num_iterations, num_policies, elapsed_time, v, policy


if __name__ == '__main__':
    results = []
    test_folder = "test"

    test_files = [f for f in os.listdir(test_folder) if f.startswith("test_case_")]
    sorted_test_files = sorted(test_files, key=lambda x: int(x.split('_')[2]))

    # Travel through all the test case in the test folder and return test result
    for filename in sorted_test_files:
        filepath = os.path.join(test_folder, filename)
        m, n, state_num, num_iterations, num_policies, elapsed_time, v, policy = run_value_iteration_for_file(filepath)

        solution_filename = filename.replace("test_case", "solution")
        solution_filepath = os.path.join(test_folder, solution_filename)
        save_solution_to_file(solution_filepath, v, policy)

        results.append((filename, m, n, num_iterations, num_policies, elapsed_time))

        print(f"Processed {filename}:")
        print(f"  Grid Dimensions: {m}x{n}")
        print(f"  Number of States: {state_num}")
        print(f"  Number of Iterations: {num_iterations}")
        print(f"  Number of Policies: {num_policies}")
        print(f"  Elapsed Time: {elapsed_time:.2f} seconds")
        print("-" * 50)

    # Save results to a table (e.g., CSV format)
    with open("results_table.csv", 'w') as f:
        f.write("Filename,m,n,Num Iterations,Num Policies,Elapsed Time\n")
        for row in results:
            f.write(",".join(map(str, row)) + "\n")
