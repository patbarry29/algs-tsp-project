import tsplib95
import numpy as np
from collections import defaultdict

# Function to create the distance matrix from the TSP problem
def create_distance_matrix(problem):
    nodes = list(problem.get_nodes())
    num_nodes = problem.dimension
    distance_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                distance_matrix[i][j] = problem.get_weight(nodes[i], nodes[j])
    return distance_matrix

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

    # Output the ordered sequence of nodes to visit and the total cost
    print("Sequence:", route[:step_counter + 2])
    print("Cost:", total_cost)

# Driver Code
if __name__ == "__main__":
    file_path = 'ALL_tsp/bayg29.tsp'
    problem = tsplib95.load(file_path)
    distance_matrix = create_distance_matrix(problem)
    print("Distance Matrix:")
    print(distance_matrix)
    find_min_route(distance_matrix)
