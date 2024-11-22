import numpy as np
from save_tsp_file import save_matrix_file
from utils.add_sparsity import add_sparsity
from utils.random_edge_weights import generate_random_weights, generate_weight_distribution

def generate_atsp(n, dim_size=100, sparsity=0, random_weights=False, weight_dist=None, int_weights=False):
    """
    Generate an Asymmetric Traveling Salesperson Problem (ATSP) instance with configurable options.

    Parameters:
        n (int): Number of cities (nodes) in the ATSP instance.
        dim_size (int): The maximum possible weight for edges in the distance matrix.
                        Edge weights will be integers in the range [1, dim_size).
        sparsity (float): Level of sparsity in the graph (0 for fully connected, up to 1 for minimal connectivity).
        random_weights (bool): If True, random weights will replace the default integer weights.
        weight_dist (dict or None): Configuration for generating weights using a specific statistical distribution.
                                    Should be a dictionary with 'type' (e.g., "normal", "uniform") and other parameters
                                    (e.g., {"mean": 50, "std": 10} for a normal distribution).
    """
    # Generate a random asymmetric distance matrix with integers in [1, dim_size)
    distance_matrix = np.random.uniform(1, dim_size, size=(n, n))
    if int_weights:
        distance_matrix = np.random.randint(1, dim_size, size=(n, n))
    np.fill_diagonal(distance_matrix, 0)  # Ensure no self-loops

    if random_weights:
        distance_matrix = generate_random_weights(distance_matrix)

    if weight_dist:
        distance_matrix = generate_weight_distribution(distance_matrix, weight_dist['type'], params=weight_dist)

    if sparsity > 0:
        distance_matrix = add_sparsity(distance_matrix, sparsity, symmetric=False)

    # Save to file
    save_matrix_file(matrix=distance_matrix, filename="random_atsp.atsp", type="ATSP")

if __name__ == "__main__":
    generate_atsp(n=4, dim_size=10, sparsity=0.8, random_weights=True,
                  weight_dist={"type": "normal", "mean": 30, "std": 5})
