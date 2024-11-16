import math
import tsplib95
import itertools
import time
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from create_distance_matrix import create_distance_matrix



def calculate_tour_distance(tour, distance_matrix, curr_best):
    """
    Calculate the total distance of a tour using the distance matrix.
    """
    total_distance = 0
    for i in range(len(tour)):
        # Get the weight between current city and next city (wrap around to start for last city)
        current_city = tour[i]
        next_city = tour[(i + 1) % len(tour)]
        total_distance += distance_matrix[current_city][next_city]
        if total_distance > curr_best:
            return total_distance
    return total_distance

def print_loading_bar(i, total):
    progress = (i / total) * 100
    sys.stdout.write(f"\rProgress: [{'#' * int(progress // 2)}{'.' * (50 - int(progress // 2))}] {progress:.2f}%")
    sys.stdout.flush()

def solve_tsp_brute_force(problem):
    """
    Solve TSP using brute force approach with pre-computed distance matrix.
    """
    # Create distance matrix
    distance_matrix = create_distance_matrix(problem)

    # Get number of nodes
    num_nodes = problem.dimension
    if num_nodes > 15:
      raise ValueError("It is unfeasible to run brute force on this problem.")

    # Use indices for cities (0 to num_nodes-1)
    start_city = 0
    other_cities = list(range(1, num_nodes))

    best_tour = None
    best_distance = sys.float_info.max

    # Try all possible permutations of other cities
    total_permutations = math.factorial(num_nodes - 1)
    i = 0
    for perm in itertools.permutations(other_cities):
      # Create complete tour by adding start city at beginning and end
      current_tour = [start_city] + list(perm) + [start_city]

      # Calculate total distance using the distance matrix
      current_distance = calculate_tour_distance(current_tour, distance_matrix, best_distance)

      i += 1
      # Print loading bar
      print_loading_bar(i, total_permutations)

      # Update best tour if current tour is shorter
      if current_distance < best_distance:
        best_distance = current_distance
        best_tour = current_tour

    print()

    # Convert indices back to actual node numbers if needed
    if best_tour:
        nodes = list(problem.get_nodes())
        best_tour = [nodes[i] for i in best_tour]

    return best_tour, best_distance

if __name__ == "__main__":
    # Example usage with a TSPLIB problem
    problem = tsplib95.load('random_atsp.atsp')

    # Solve the problem
    start = time.time()
    best_tour, best_distance = solve_tsp_brute_force(problem)
    end = time.time()

    # Print results
    print(f"Best tour found: {best_tour}")
    print(f"Tour distance: {best_distance}")
    print(f"Time Taken: {end-start}")