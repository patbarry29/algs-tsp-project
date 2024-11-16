import matplotlib.pyplot as plt
import tsplib95
from create_distance_matrix import create_distance_matrix
from generate_tsp import generate_tsp

# Input: A function that solves the TSP problem (this function should return the route and total cost)
def plot_cost_vs_cities(tsp_solver):
    num_cities_list = list(range(5, 1000, 100))  # Number of cities from 5 to 50 in steps of 5
    costs = []  # To store the total cost for each TSP problem

    for num_cities in num_cities_list:
        # Generate TSP problem
        cities = generate_tsp(num_cities, dim_size=100)

        # Load the problem instance using TSPLIB
        problem = tsplib95.load("random_tsp.tsp")

        # Generate distance matrix
        distance_matrix = create_distance_matrix(problem)

        # Solve the TSP using the provided algorithm
        _, total_cost = tsp_solver(distance_matrix)
        costs.append(total_cost)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(num_cities_list, costs, marker="o", linestyle="-", label="TSP Cost")
    plt.xlabel("Number of Cities")
    plt.ylabel("Total Cost")
    plt.title("Cost vs. Number of Cities in TSP")
    plt.grid(True)
    plt.legend()
    plt.show()
