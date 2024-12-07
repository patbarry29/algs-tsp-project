import matplotlib.pyplot as plt
import tracemalloc
import numpy as np
import tsplib95
from generate_tsp import generate_tsp
from create_distance_matrix import create_distance_matrix

def plot_cost_vs_cities_with_memory(tsp_solver):
    np.random.seed(1)  # Set seed for reproducibility

    # Define the range of city counts
    num_cities_list = list(range(1, 100, 10))
    print(num_cities_list)
    costs = []  # To store the total cost for each TSP problem
    memory_usages = []  # To store the peak memory usage for each problem

    for num_cities in num_cities_list:
        # Generate TSP problem with specified number of cities
        cities = generate_tsp(n=num_cities)

        # Load the problem instance using TSPLIB
        problem = tsplib95.load("random_tsp.tsp")

        # Generate distance matrix
        distance_matrix = create_distance_matrix(problem)

        # Measure memory usage and solve the TSP
        tracemalloc.start()
        try:
            _, total_cost = tsp_solver(distance_matrix)
            costs.append(total_cost)
        except Exception as e:
            print(f"Error solving TSP for {num_cities} cities: {e}")
            costs.append(None)
        # Record memory usage
        _, peak_memory = tracemalloc.get_traced_memory()
        memory_usages.append(peak_memory / 1024)  # Convert to KB
        tracemalloc.stop()

    # Plot the results
    plt.figure(figsize=(10, 6))

    # Plot memory usage vs number of cities
    plt.plot(
        num_cities_list,
        memory_usages,
        marker="o", linestyle="-", label="Peak Memory Usage (KB)"
    )
    plt.xlabel("Number of Cities")
    plt.ylabel("Memory Usage (KB)")
    plt.title("Memory Usage vs. Number of Cities in TSP")
    plt.grid(True)
    plt.legend()
    plt.show()
