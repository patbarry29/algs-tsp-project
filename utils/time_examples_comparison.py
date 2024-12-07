import time
import matplotlib.pyplot as plt
import numpy as np
import tsplib95

from generate_tsp import generate_tsp
from create_distance_matrix import create_distance_matrix

def plot_time_vs_cities(tsp_solver):
    """
    Plot the execution time of the TSP solver as the number of cities increases.

    Parameters:
    tsp_solver (function): A function that solves the TSP problem and returns (route, total_cost).
    """
    np.random.seed(1)  # Set seed for reproducibility
    num_cities_list = list(range(5, 100, 10))  # Number of cities from 5 to 100 in steps of 10
    times = []  # To store the execution time for each TSP problem

    for num_cities in num_cities_list:
        try:
            # Generate TSP problem
            cities = generate_tsp(num_cities, dim_size=100)

            # Load the problem instance using TSPLIB
            problem = tsplib95.load("random_tsp.tsp")

            # Generate distance matrix
            distance_matrix = create_distance_matrix(problem)

            # Measure execution time for the TSP solver
            start_time = time.time()
            tsp_solver(distance_matrix)
            end_time = time.time()

            times.append(end_time - start_time)
        except Exception as e:
            print(f"Error solving TSP for {num_cities} cities: {e}")
            times.append(None)

    # Filter out None values for plotting
    filtered_cities = [n for n, t in zip(num_cities_list, times) if t is not None]
    filtered_times = [t for t in times if t is not None]

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(
        filtered_cities,
        filtered_times,
        marker="o",
        linestyle="-",
        label="Execution Time"
    )
    plt.xlabel("Number of Cities")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Execution Time vs. Number of Cities in TSP")
    plt.grid(True)
    plt.legend()
    plt.show()
