import matplotlib.pyplot as plt
import tsplib95
import numpy as np
from create_distance_matrix import create_distance_matrix
from generate_tsp import generate_tsp

# Input: A function that solves the TSP problem (this function should return the route and total cost)
def plot_cost_vs_cities(tsp_solvers, solver_names):
    if len(tsp_solvers) != len(solver_names):
        raise ValueError("Number of solvers must match number of solver names")

    np.random.seed(np.random.randint(10))
    num_cities_list = list(range(1,10,1))
    costs_per_solver = []

    for num_cities in num_cities_list:
        # Generate one TSP problem for all solvers
        cities = generate_tsp(num_cities, dim_size=100)
        problem = tsplib95.load("random_tsp.tsp")
        distance_matrix = create_distance_matrix(problem)

        # For first iteration, initialize the lists
        if num_cities == num_cities_list[0]:
            costs_per_solver = [[] for _ in range(len(tsp_solvers))]

        # Run each solver on the same problem
        for i, solver in enumerate(tsp_solvers):
            _, total_cost = solver(distance_matrix)
            costs_per_solver[i].append(total_cost)
            print(f"{solver_names[i]} cost: {total_cost}")
            print()

    # Plot the results
    plt.figure(figsize=(10, 6))
    for i, costs in enumerate(costs_per_solver):
        plt.plot(num_cities_list, costs, marker="o", linestyle="-", label=solver_names[i])

    plt.xlabel("Number of Cities")
    plt.ylabel("Total Cost")
    plt.title("Cost vs. Number of Cities in TSP")
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