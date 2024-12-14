import os
import sys
import time
import tsplib95

# Add the parent directory to the system path to import modules from there
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.create_distance_matrix import create_distance_matrix

def lin_kernighan(tsp_matrix):
    """
    Solve the TSP using the Lin-Kernighan heuristic to find an approximate minimum cost path.

    Parameters:
    tsp_matrix (np.ndarray): The distance matrix for the TSP.

    Returns:
    tuple: A list of the route sequence and the total cost.
    """
    num_nodes = len(tsp_matrix)  # Number of nodes in the TSP
    # Initialize a tour starting from node 0
    current_tour = list(range(num_nodes))
    best_tour = current_tour[:]  # Copy of the current tour as the best tour
    best_cost = calculate_cost(best_tour, tsp_matrix)  # Calculate the cost of the initial tour

    # Implement the Lin-Kernighan heuristic
    improved = True
    while improved:
        improved = False
        # Try to improve the tour using k-opt moves
        for k in range(2, num_nodes):
            new_tour = perform_k_opt_move(current_tour, k, tsp_matrix)  # Perform a k-opt move
            new_cost = calculate_cost(new_tour, tsp_matrix)  # Calculate the cost of the new tour
            if new_cost < best_cost:  # If the new tour is better, update the best tour
                best_tour = new_tour[:]
                best_cost = new_cost
                improved = True  # Set improved to True to continue the loop
                break  # Exit the loop to start over with the improved tour

    # Convert tour indices to start from 1
    best_tour = [city + 1 for city in best_tour]

    return best_tour, best_cost  # Return the best tour and its cost

def calculate_cost(tour, tsp_matrix):
    """Calculate the total cost of a given tour."""
    # Sum the distances between consecutive nodes in the tour
    return sum(tsp_matrix[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))

def perform_k_opt_move(tour, k, tsp_matrix):
    """Perform a 2-opt move on the tour."""
    best_tour = tour[:]
    best_cost = calculate_cost(tour, tsp_matrix)
    
    for i in range(len(tour) - 1):
        for j in range(i + 2, len(tour)):
            if j - i == 1: 
                continue  # Skip adjacent edges
            # Reverse the segment between i and j
            new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
            new_cost = calculate_cost(new_tour, tsp_matrix)
            if new_cost < best_cost:
                best_tour = new_tour
                best_cost = new_cost
    
    return best_tour

# Driver Code
if __name__ == "__main__":
    # Load a TSP problem from a file
    problem = tsplib95.load('data/ALL_tsp/burma14.tsp')
    # Create a distance matrix from the problem
    distance_matrix = create_distance_matrix(problem)

    start_time = time.time()  # Record the start time
    # Solve the TSP using the Lin-Kernighan heuristic
    route, total_cost = lin_kernighan(distance_matrix)
    running_time = time.time() - start_time  # Calculate the running time

    # Print the results
    print("Sequence:", route)
    print("Cost:", total_cost)
    print(f"Running time: {running_time:.6f} seconds") 