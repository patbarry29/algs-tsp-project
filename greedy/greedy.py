import tsplib95
from collections import defaultdict
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from create_distance_matrix import create_distance_matrix
from utils.cost_examples_comparison import plot_cost_vs_cities
from utils.performance import compute_cpu_usage
from utils.get_opt_cost import get_optimal_cost
from data.opt_cost import tsp as opt_sol

def relative_error(optimal_cost, total_cost):
    # Create a dictionary for quick lookup
    optimal_cost_dict = dict(zip(opt_sol.data["Instance"], opt_sol.data["OptimalCost"]))

    # Get the optimal cost for bayg29
    optimal_cost = optimal_cost_dict["bayg29"]
    print(f"Total Cost (Recomputed): {optimal_cost}")
    return (total_cost - optimal_cost) / optimal_cost * 100

# Adapted from https://www.geeksforgeeks.org/travelling-salesman-problem-greedy-approach/
# Function to find the minimum cost path for all paths using a greedy approach
def find_min_route(tsp_matrix):
    num_nodes = len(tsp_matrix)
    total_cost = 0
    step_counter = 0
    current_city = 0
    min_distance = float('inf')
    visited_cities = defaultdict(int)

    # Starting from the first city (index 0)
    visited_cities[0] = 1
    route = [0] * (num_nodes + 1)  # Ensure the route list is large enough
    route[0] = 1  # Start from the first city (index 0)

    # Traverse the distance matrix
    while step_counter < num_nodes - 1:
        for next_city in range(num_nodes):
            # If this path is unvisited and cost is less, update min_distance
            if next_city != current_city and not visited_cities[next_city]:
                if tsp_matrix[current_city][next_city] < min_distance:
                    min_distance = tsp_matrix[current_city][next_city]
                    route[step_counter + 1] = next_city + 1

        # Update the total cost and reset for the next iteration
        total_cost += min_distance
        min_distance = float('inf')
        visited_cities[route[step_counter + 1] - 1] = 1
        current_city = route[step_counter + 1] - 1
        step_counter += 1

    # Complete the tour by returning to the starting city
    last_city = route[step_counter] - 1
    for next_city in range(num_nodes):
        if last_city != next_city and tsp_matrix[last_city][next_city] < min_distance:
            min_distance = tsp_matrix[last_city][next_city]
            route[step_counter + 1] = next_city + 1

    total_cost += min_distance

    # Add the return to the starting city to the route
    route[step_counter + 1] = 1

    return route[:step_counter + 2], total_cost

# Driver Code
if __name__ == "__main__":
    data = "bayg29"
    # Load the problem instance using TSPLIB
    problem = tsplib95.load(f'../data/ALL_tsp/{data}.tsp')

    # Generate the distance matrix
    distance_matrix = create_distance_matrix(problem)
    # print("Distance Matrix:")
    # print(distance_matrix)

    # Measure CPU usage and execution time
    result, running_time, cpu_usage = compute_cpu_usage(find_min_route, distance_matrix)

    # Extract results
    route, total_cost = result

    # Output the solution
    print("Sequence:", route)
    print("Cost:", total_cost)

    print(f"Running time: {running_time:.4f} seconds")
    print(f"CPU Usage: {cpu_usage:.4f}%")

    # Add TSPLIB optimal solutions for comparison if known
    optimal_cost = get_optimal_cost(opt_sol.data, data)
    if optimal_cost is not None:
        relative_error = (total_cost - optimal_cost) / optimal_cost * 100
        print(f"Optimal Cost: {optimal_cost}")
        print(f"Relative Error: {relative_error:.2f}%")

    # Plot the cost vs. number of cities
    plot_cost_vs_cities(find_min_route)