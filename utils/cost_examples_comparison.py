# IMPORTS
import tsplib95
import matplotlib.pyplot as plt
import numpy as np

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from utils.create_distance_matrix import create_distance_matrix
from utils.generate_tsp import generate_tsp

from randomized.randomized import randomized
from ant_colony.ant_colony import ant_colony
from brute_force.brute_force import brute_force
from greedy.greedy import find_min_route



# Input: A function that solves the TSP problem (this function should return the route and total cost)
def plot_cost_vs_cities(tsp_solvers, solver_names, problem_sizes):
    if len(tsp_solvers) != len(solver_names):
        raise ValueError("Number of solvers must match number of solver names")

    np.random.seed(np.random.randint(10))
    avg_costs_per_solver = []

    for num_cities in problem_sizes:
        # For first iteration, initialize the lists
        if num_cities == problem_sizes[0]:
            avg_costs_per_solver = [[] for _ in range(len(tsp_solvers))]

        for _ in range(10):  # Create 10 random problems for each number of cities
            # Generate one TSP problem for all solvers
            cities = generate_tsp(num_cities, dim_size=100)
            problem = tsplib95.load("data/random/tsp/random_tsp.tsp")
            distance_matrix = create_distance_matrix(problem)

            # Run each solver on the same problem
            for i, solver in enumerate(tsp_solvers):
                _, total_cost = solver(distance_matrix)
                avg_costs_per_solver[i].append(total_cost)
                print(f"{solver_names[i]} cost: {total_cost}")
                print()

    # Calculate the average costs
    avg_costs_per_solver = [[np.mean(avg_costs_per_solver[i][j:j+10]) for j in range(0, len(avg_costs_per_solver[i]), 10)] for i in range(len(tsp_solvers))]

    # Plot the results
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^', 'd', '*']  # Add more markers if needed
    colors = plt.cm.tab10.colors  # Use a colormap for distinct colors
    for i, avg_costs in enumerate(avg_costs_per_solver):
        jittered_x = [x + i*(0.02) for x in problem_sizes]
        plt.plot(jittered_x, avg_costs, marker=markers[i % len(markers)],
                color=colors[i % len(colors)], linestyle="-", label=solver_names[i])

    plt.xlabel("Number of Cities")
    plt.ylabel("Average Total Cost")
    plt.title("Average Cost vs. Number of Cities in TSP")
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
    algorithms = [ant_colony, find_min_route, randomized]
    # np.random.seed(1)
    algorithms_names = ['ACO', 'Greedy', 'Randomized']
    problem_sizes = list(range(10,101,10))
    plot_cost_vs_cities(algorithms, algorithms_names, problem_sizes)