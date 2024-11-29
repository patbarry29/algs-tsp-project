import tsplib95
import time
import numpy as np
from tqdm import tqdm

from helpers import *

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.generate_tsp import generate_tsp
from utils.create_distance_matrix import create_distance_matrix
from utils.generate_atsp import generate_atsp



def ant_colony(distance_matrix, params={}):
    # first fill distance matrix diagonal with inf
    np.fill_diagonal(distance_matrix, np.inf)
    epsilon = 1e-10
    distance_matrix[distance_matrix == 0] = epsilon

    alpha = params.get('alpha', 1)   # exploitation param
    beta = params.get('beta', 3)    # exploration param
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
    problem = tsplib95.load('data/ALL_atsp/rbg323.atsp')
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
