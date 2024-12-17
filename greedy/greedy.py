import time

import tsplib95
from collections import defaultdict
from utils.create_distance_matrix import create_distance_matrix

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
    data = "berlin52"
    # data = None
    problem = tsplib95.load(f'../data/ALL_tsp/{data}.tsp')
    # problem = tsplib95.load('../random_tsp.tsp')

    distance_matrix = create_distance_matrix(problem)
    # print(f"Distance Matrix: \n {distance_matrix}")

    start_time = time.time()
    route, total_cost = greedy(distance_matrix)
    running_time = time.time() - start_time

    print("Sequence:", route)
    print("Cost:", total_cost)
    print(f"Running time: {running_time:.6f} seconds")




