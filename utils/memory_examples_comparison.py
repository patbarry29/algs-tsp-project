# IMPORTS
import tsplib95
import matplotlib.pyplot as plt
import numpy as np
import psutil


from branch_and_bound.branch_and_bound import branch_and_bound
from greedy.greedy import greedy
from lin_kernighan.lin_kernighan import lin_kernighan
from randomized.randomized import randomized
from dynamic_programming
# from ant_colony.ant_colony import ant_colony
from utils.create_distance_matrix import create_distance_matrix
from utils.generate_tsp import generate_tsp
from brute_force.brute_force import brute_force


# Input: A function that solves the TSP problem (this function should return the route and total cost)
def plot_algs_vs_problem_size(tsp_solvers, solver_names, problem_sizes):
    if len(tsp_solvers) != len(solver_names):
        raise ValueError("Number of solvers must match number of solver names")

    np.random.seed(np.random.randint(10))
    avg_metrics_per_solver = []

    for num_cities in problem_sizes:
        # For first iteration, initialize the lists
        if num_cities == problem_sizes[0]:
            avg_metrics_per_solver = [[] for _ in range(len(tsp_solvers))]

        for _ in range(10):  # Create 10 random problems for each number of cities
            # Generate one TSP problem for all solvers
            cities = generate_tsp(num_cities, dim_size=100)
            problem = tsplib95.load("data/random/tsp/random_tsp.tsp")
            distance_matrix = create_distance_matrix(problem)

            # Run each solver on the same problem
            for i, solver in enumerate(tsp_solvers):
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024  # Memory in KB
                _ = solver(distance_matrix)
                memory_after = process.memory_info().rss / 1024  # Memory in KB
                memory_used = memory_after - memory_before
                avg_metrics_per_solver[i].append(memory_used)
                print(f"{solver_names[i]} memory used: {memory_used} KB")
                print()

    # Calculate the average metrics
    avg_metrics_per_solver = [[np.mean(avg_metrics_per_solver[i][j:j + 10])
                               for j in range(0, len(avg_metrics_per_solver[i]), 10)]
                              for i in range(len(tsp_solvers))]

    # Plot the results
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^', 'd', '*']  # Add more markers if needed
    colors = plt.cm.tab10.colors  # Use a colormap for distinct colors

    for i, avg_metrics in enumerate(avg_metrics_per_solver):
        jittered_x = [x + i * (0.02) for x in problem_sizes]
        plt.plot(jittered_x, avg_metrics, marker=markers[i % len(markers)],
                 color=colors[i % len(colors)], label=solver_names[i])

    plt.xlabel("Number of Cities")
    plt.ylabel("Average Memory Used (KB)")
    plt.title("Average Memory Used vs. Number of Cities in TSP")
    plt.grid(True)
    plt.legend()

    plt.show()
    plt.style.use('seaborn')  # Use a style that enhances visibility
    plt.rcParams['axes.facecolor'] = 'white'  # Set white background
    plt.rcParams['axes.grid'] = True  # Enable grid
    plt.rcParams['grid.alpha'] = 0.3  # Make grid subtle
    for line in plt.gca().lines:
        line.set_alpha(0.8)  # Set line transparency
        line.set_linewidth(2)  # Make lines thicker


if __name__ == '__main__':
    algorithms = [brute_force, randomized, branch_and_bound, greedy, lin_kernighan]
    algorithms_names = ['Brute Force', 'Randomised', 'Branch and Bound', 'Greedy', 'Lin-Kernighan']
    problem_sizes = list(range(3, 10, 1))
    plot_algs_vs_problem_size(algorithms, algorithms_names, problem_sizes)