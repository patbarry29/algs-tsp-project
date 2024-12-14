import tsplib95
import time
import numpy as np
from tqdm import tqdm

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.generate_tsp import generate_tsp
from utils.create_distance_matrix import create_distance_matrix
from utils.generate_atsp import generate_atsp



def get_start_points(num_ants, cities):
    """
    Generate random start points for the ants.

    Parameters:
    num_ants (int): Number of ants.
    cities (array-like): List or array of cities.

    Returns:
    numpy.ndarray: Array of start points for each ant.
    """
    start_points = np.random.randint(0, len(cities), num_ants)
    return start_points


def create_probabilities(combined_matrix, tours):
    # Create mask of unvisited cities for each ant
    num_cities = combined_matrix.shape[0]
    visited_mask = np.zeros((tours.shape[0], num_cities), dtype=bool)
    np.put_along_axis(visited_mask, tours, True, axis=1)

    # Get current positions
    current_positions = tours[:,-1]

    # Get probabilities only for unvisited cities
    probs = combined_matrix[current_positions]
    probs[visited_mask] = 0  # Zero out visited cities
    return probs

def calculate_selection_probs(combined_matrix, tours):
    # Get raw probabilities
    probs = create_probabilities(combined_matrix, tours)

    # Normalize and return cumulative probabilities
    row_sums = np.sum(probs, axis=1)[:,None]
    normalised_matrix = probs/row_sums
    return np.cumsum(normalised_matrix, axis=1)


def find_next_cities(n, probs, tours):
    selection_thresholds = np.random.rand(n)
    selections = probs > selection_thresholds[:,None]
    # Handle case where no valid selection (all visited)
    no_selection = ~np.any(selections, axis=1)
    selections[no_selection, -1] = True  # Force select last option if none found
    return np.argmax(selections, axis=1)


def build_tours(matrix, size, tours, num_ants):
    while tours.shape[1] < size:
        selection_probs = calculate_selection_probs(matrix, tours)
        next_cities = find_next_cities(num_ants, selection_probs, tours)
        tours = np.append(tours, next_cities[:,None], axis=1)
    tours = np.append(tours, tours[:,0][:,None], axis=1)

    return tours


def update_pheromones(pheromones, evaporation_rate, tour_lengths, tours):
    pheromones *= (1-evaporation_rate)
    Q = 10
    pheromone_deposits = Q/tour_lengths

    for i, tour in enumerate(tours):
        for j in range(len(tour)-1):
            pheromones[tour[j]][tour[j+1]] += pheromone_deposits[i]
    return pheromones


def ant_colony(distance_matrix, params={}):
    # first fill distance matrix diagonal with inf
    np.fill_diagonal(distance_matrix, np.inf)
    epsilon = 1e-10
    distance_matrix[distance_matrix == 0] = epsilon

    alpha = params.get('alpha', 1)   # exploitation param
    beta = params.get('beta', 4)    # exploration param
    num_iterations = params.get('num_iterations', 100)
    evaporation_rate = params.get('evaporation_rate', 0.1)
    initial_pheromone_level = params.get('initial_pheromone_level', 0.1)
    num_ants = params.get('num_ants', 10)

    # create pheromone matrix
    pheromones = np.full_like(distance_matrix, initial_pheromone_level)

    # randomly assign a start city to each ant
    cities = np.arange(distance_matrix.shape[0])
    start_points = get_start_points(num_ants, cities)

    one_over_distance = (1/distance_matrix)

    iteration = 0
    best_tour = None
    best_distance = float('inf')
    best_distances = []

    # Initialize progress bar
    if not params.get('mute_output', False):
        progress_bar = tqdm(total=num_iterations, desc="Progress", unit="iteration", dynamic_ncols=True)

    while iteration < num_iterations:
        tours = start_points.reshape(-1, 1)
        combined_matrix = ((one_over_distance)**beta)*(pheromones**alpha)
        tours = build_tours(combined_matrix, len(cities), tours, num_ants)

        tour_lengths = np.sum([distance_matrix[tours[:, i], tours[:, i+1]] for i in range(len(cities))], axis=0)

        pheromones = update_pheromones(pheromones, evaporation_rate, tour_lengths, tours)

        iteration_best = np.min(tour_lengths)
        if iteration_best < best_distance:
            best_distance = iteration_best
            best_tour = tours[np.argmin(tour_lengths)]

        best_distances.append(iteration_best)

        # Update progress bar and statistics
        if not params.get('mute_output', False):
            progress_bar.set_postfix(best_distance=best_distance, iteration_best=iteration_best)
            progress_bar.update(1)

        iteration += 1

    if not params.get('mute_output', False):
        progress_bar.close()

    if params.get('return_all', False):
        return best_tour+1, np.round(best_distance,2), best_distances
    return best_tour+1, np.round(best_distance,2)


if __name__ == "__main__":
    # generate_tsp(n=100)
    # problem = tsplib95.load('data/random/tsp/random_tsp.tsp')
    problem = tsplib95.load('data/ALL_tsp/bier127.tsp')
    # problem = tsplib95.load('data/random/atsp/random_atsp.atsp')

    distance_matrix = create_distance_matrix(problem)

    params = {
        'num_iterations': 100
    }
    # Solve the problem
    start = time.time()
    best_tour, best_distance = ant_colony(distance_matrix, params)
    end = time.time()

    # Print results
    print(f"Best tour found: {best_tour}")
    print(f"Tour distance: {best_distance}")
    print(f"Time Taken: {end-start}")
