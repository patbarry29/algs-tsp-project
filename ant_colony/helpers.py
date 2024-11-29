
import numpy as np


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

