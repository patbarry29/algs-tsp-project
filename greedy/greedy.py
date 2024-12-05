import tsplib95
from collections import defaultdict
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from create_distance_matrix import create_distance_matrix
from utils.cost_examples_comparison import plot_cost_vs_cities
from utils.memory_examples_comparison import plot_cost_vs_cities_with_memory
from utils.performance import compute_cpu_usage
from utils.get_opt_cost import get_optimal_cost
from data.opt_cost import tsp as opt_sol
from utils.time_examples_comparison import plot_time_vs_cities


# Adapted from https://www.geeksforgeeks.org/travelling-salesman-problem-greedy-approach/
# Function to find the minimum cost path for all paths using a greedy approach
def greedy(tsp_matrix):
    """
    Solve the TSP using a greedy approach to find an approximate minimum cost path.

    Parameters:
    tsp_matrix (np.ndarray): The distance matrix for the TSP.

    Returns:
    tuple: A list of the route sequence and the total cost.
    """
    num_nodes = len(tsp_matrix)
    total_cost = 0
    step_counter = 0
    current_city = 0
    min_distance = float('inf')
    visited_cities = defaultdict(int)

    visited_cities[0] = 1       # Mark the starting city as visited
    route = [0] * (num_nodes + 1)  # Route array with space for return to start
    route[0] = 1  # Start from the first city (index 0)

    # Loop to visit all cities
    while step_counter < num_nodes - 1:
        for next_city in range(num_nodes):
            if next_city != current_city and not visited_cities[next_city]: # If the city is unvisited
                if tsp_matrix[current_city][next_city] < min_distance: # If the cost is less than the minimum distance
                    min_distance = tsp_matrix[current_city][next_city]  # Update the minimum distance
                    route[step_counter + 1] = next_city + 1 # Update the route

        total_cost += min_distance  # Add the minimum distance to the total cost
        # print(total_cost)
        min_distance = float('inf') # Reset the minimum distance
        visited_cities[route[step_counter + 1] - 1] = 1 # Mark the city as visited
        current_city = route[step_counter + 1] - 1 # Update the current city
        step_counter += 1

    # Return to the starting city
    last_city = route[step_counter] - 1 # Get the last city visited
    total_cost += tsp_matrix[last_city][0]   # Update the total cost
    route[step_counter + 1] = 1  # Update the route
    # print(f"Last city visited: {last_city}")
    # print(f"Returning to City 0 with distance: {tsp_matrix[last_city][0]}")

    return route[:step_counter + 2], total_cost

# Driver Code
if __name__ == "__main__":
    # data = "br17"
    data = None
    # Load the problem instance using TSPLIB
    # problem = tsplib95.load(f'../data/ALL_atsp/{data}.atsp')
    problem = tsplib95.load('../random_tsp.tsp')

    # Generate the distance matrix
    distance_matrix = create_distance_matrix(problem)
    print("Distance Matrix:")
    print(distance_matrix)

    # Measure CPU usage and execution time
    result, running_time, cpu_usage = compute_cpu_usage(greedy, distance_matrix)

    # Extract results
    route, total_cost = result

    # Output the solution
    print("Sequence:", route)
    print("Cost:", total_cost)

    print(f"Running time: {running_time:.6f} seconds")
    print(f"CPU Usage: {cpu_usage:.4f}%")

    # Add TSPLIB optimal solutions for comparison if known
    optimal_cost = get_optimal_cost(opt_sol.data, data)
    if optimal_cost is not None:
        relative_error = (total_cost - optimal_cost) / optimal_cost * 100
        print(f"Optimal Cost: {optimal_cost}")
        print(f"Relative Error: {relative_error:.2f}%")

    # Plot the cost vs. number of cities
    # plot_cost_vs_cities(greedy)

    # Plot the cost vs. number of cities with memory usage
    # plot_cost_vs_cities_with_memory(greedy)

    # Plot the time vs. number of cities
    plot_time_vs_cities(greedy)