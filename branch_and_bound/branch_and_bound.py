import sys
import os
import tsplib95
import time

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the utils directory to sys.path
utils_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(utils_dir)

# Import the modules
from utils.create_distance_matrix import create_distance_matrix
from utils.generate_atsp import generate_atsp
def calculate_tour_distance(tour, distance_matrix):
    """
    Calculate the total distance of a tour using the distance matrix.
    """
    total_distance = 0
    for i in range(len(tour) - 1):
        current_city = tour[i]
        next_city = tour[(i + 1) % len(tour)]
        total_distance += distance_matrix[current_city][next_city]
    return total_distance

def print_loading_bar(i, total):
    progress = (i / total) * 100
    sys.stdout.write(f"\rProgress: [{'#' * int(progress // 2)}{'.' * (50 - int(progress // 2))}] {progress:.2f}%")
    sys.stdout.flush()

def branch_and_bound(distance_matrix):
    """
    Solve TSP using Branch and Bound approach with pre-computed distance matrix.
    """
    num_nodes = distance_matrix.shape[0]
    best_tour = None
    best_distance = float('inf')

    # Initialize the minimum cost matrix
    def calculate_bound(visited, current_distance):
        """
        Calculate a lower bound for the current state.
        """
        bound = current_distance
        for i in range(num_nodes):
            if i not in visited:
                min_edge = min(distance_matrix[i][j] for j in range(num_nodes) if j != i)
                bound += min_edge
        return bound

    def dfs(current_city, current_tour, visited, current_distance):
        """
        Perform DFS for Branch and Bound.
        """
        nonlocal best_tour, best_distance

        if len(current_tour) == num_nodes:
            # Add return to start city
            current_distance += distance_matrix[current_city][current_tour[0]]
            if current_distance < best_distance:
                best_distance = current_distance
                best_tour = current_tour + [current_tour[0]]
            return

        for next_city in range(num_nodes):
            if next_city not in visited:
                next_distance = current_distance + distance_matrix[current_city][next_city]
                bound = calculate_bound(visited | {next_city}, next_distance)

                # Prune if bound is worse than current best
                if bound < best_distance:
                    dfs(next_city, current_tour + [next_city], visited | {next_city}, next_distance)

    # Start the search from the first city
    start_city = 0
    dfs(start_city, [start_city], {start_city}, 0)

    if best_tour:
        best_tour = [i + 1 for i in best_tour]

    return best_tour, round(best_distance, 2)

if __name__ == "__main__":
    # Example usage with a TSPLIB problem
    generate_atsp(n=20)
    problem = tsplib95.load('data/random/atsp/random_atsp.atsp')
    distance_matrix = create_distance_matrix(problem)

    # Solve the problem
    start = time.time()
    best_tour, best_distance = branch_and_bound(distance_matrix)
    end = time.time()

    # Print results
    print(f"Best tour found: {best_tour}")
    print(f"Tour distance: {best_distance}")
    print(f"Time Taken: {end - start}")
