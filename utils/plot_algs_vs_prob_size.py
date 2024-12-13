# IMPORTS
import tsplib95
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import time
import psutil



current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)


from genetic.genetic import genetic
from branch_and_bound.branch_and_bound import branch_and_bound
from greedy.greedy import greedy
from lin_kernighan.lin_kernighan import lin_kernighan
from Randomised.randomised import randomised
from ant_colony.ant_colony import ant_colony
from utils.create_distance_matrix import create_distance_matrix
from utils.generate_tsp import generate_tsp
from brute_force.brute_force import brute_force
from performance import compute_cpu_usage


# Input: A function that solves the TSP problem (this function should return the route and total cost)
def plot_algs_vs_problem_size(tsp_solvers, solver_names, problem_sizes, measure='cost'):
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
                if measure == 'cost':
                    _, total_cost = solver(distance_matrix)
                    avg_metrics_per_solver[i].append(total_cost)
                    print(f"{solver_names[i]} cost: {total_cost}")
                elif measure == 'time':
                    _, running_time, _ = compute_cpu_usage(solver, distance_matrix)
                    avg_metrics_per_solver[i].append(running_time)
                    print(f"{solver_names[i]} execution time: {running_time}")
                elif measure == 'both':
                    result, running_time, _ = compute_cpu_usage(solver, distance_matrix)
                    total_cost = result[1]
                    avg_metrics_per_solver[i].append((total_cost, running_time))
                    print(f"{solver_names[i]} cost: {total_cost}, execution time: {running_time}")
                print()

    # Calculate the average metrics
    if measure == 'both':
        avg_metrics_per_solver = [[(np.mean([x[0] for x in avg_metrics_per_solver[i][j:j+10]]),
                                    np.mean([x[1] for x in avg_metrics_per_solver[i][j:j+10]]))
                                    for j in range(0, len(avg_metrics_per_solver[i]), 10)]
                                    for i in range(len(tsp_solvers))]
    else:
        avg_metrics_per_solver = [[np.mean(avg_metrics_per_solver[i][j:j+10])
                                    for j in range(0, len(avg_metrics_per_solver[i]), 10)]
                                    for i in range(len(tsp_solvers))]

    # Plot the results
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^', 'd', '*']  # Add more markers if needed
    colors = plt.cm.tab10.colors  # Use a colormap for distinct colors
    if measure == 'both':
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        for i, avg_metrics in enumerate(avg_metrics_per_solver):
            jittered_x = [x + i*(0.02) for x in problem_sizes]
            avg_costs = [x[0] for x in avg_metrics]
            avg_times = [x[1] for x in avg_metrics]
            ax1.plot(jittered_x, avg_costs, marker=markers[i % len(markers)],
                    color=colors[i % len(colors)], linestyle="-", label=f"{solver_names[i]} Cost")
            ax2.plot(jittered_x, avg_times, marker=markers[i % len(markers)],
                    color=colors[i % len(colors)], linestyle="-", label=f"{solver_names[i]} Time")
            # for j, txt in enumerate(avg_times):
            #     ax2.annotate(f"{txt:.2f}", (jittered_x[j], avg_times[j]), textcoords="offset points", xytext=(0,5), ha='center')

        ax1.set_xlabel("Number of Cities")
        ax1.set_ylabel("Average Total Cost")
        ax1.set_title("Average Cost vs. Number of Cities in TSP")
        ax1.grid(True)
        ax1.legend()

        ax2.set_xlabel("Number of Cities")
        ax2.set_ylabel("Average Execution Time")
        ax2.set_title("Average Execution Time vs. Number of Cities in TSP")
        ax2.grid(True)
        ax2.legend()
    else:
        for i, avg_metrics in enumerate(avg_metrics_per_solver):
            jittered_x = [x + i*(0.02) for x in problem_sizes]
            plt.plot(jittered_x, avg_metrics, marker=markers[i % len(markers)],
                    color=colors[i % len(colors)], label=solver_names[i])
            if measure == 'time':
                for j, txt in enumerate(avg_metrics):
                    plt.annotate(f"{txt:.2f}", (jittered_x[j], avg_metrics[j]), textcoords="offset points", xytext=(0,5), ha='center')

        plt.xlabel("Number of Cities")
        plt.ylabel("Average Total Cost" if measure == 'cost' else "Average Execution Time")
        plt.title("Average Cost vs. Number of Cities in TSP" if measure == 'cost' else "Average Execution Time vs. Number of Cities in TSP")
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
    algorithms = [brute_force, branch_and_bound]
    algorithms_names = ['Brute Force', 'BB']
    problem_sizes = list(range(3,10,1))
    plot_algs_vs_problem_size(algorithms, algorithms_names, problem_sizes, measure='time')