# IMPORTS
import numpy as np

import os
import sys
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, project_root)
from utils.add_sparsity import add_sparsity
from utils.random_edge_weights import generate_random_weights, generate_weight_distribution
from utils.save_tsp_file import save_matrix_file



def generate_tsp(n, dim_size=100, sparsity=0, random_weights=False, weight_dist=None, int_coords=False):
    """
    Generate a Traveling Salesperson Problem (TSP) instance with configurable options.

    Parameters:
        n (int): Number of cities (nodes) in the TSP instance.
        dim_size (int): The size of the 2D space in which cities are placed. Cities will have coordinates in [0, dim_size].
        sparsity (float): Level of sparsity in the graph (0 for fully connected, up to 1 for minimal connectivity).
        random_weights (bool): If True, random weights will be assigned to edges instead of using Euclidean distances.
        weight_dist (dict or None): Configuration for generating weights using a specific statistical distribution.
                                    Should be a dictionary with 'type' (e.g., "normal", "uniform") and other parameters
                                    (e.g., {"mean": 50, "std": 10} for a normal distribution).

    Returns:
        np.ndarray: An array of city coordinates with shape (n, 2).
    """
    cities = np.random.uniform(0, dim_size, size=(n, 2))
    if int_coords:
        cities = np.random.randint(0, dim_size, size=(n, 2))
    distance_matrix = None
    distance_matrix = np.array([[np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2) for x1 in cities] for x2 in cities])
    if random_weights:
        distance_matrix = generate_random_weights(distance_matrix, int_coords=int_coords)
    if weight_dist:
        distance_matrix = generate_weight_distribution(distance_matrix, weight_dist['type'], params=weight_dist)
    if sparsity > 0:
        distance_matrix = add_sparsity(distance_matrix, sparsity, symmetric=True)
    save_matrix_file(cities, distance_matrix)
    return cities

if __name__ == "__main__":
    cities = generate_tsp(n=10, dim_size=10, int_coords=True, sparsity=0.4)