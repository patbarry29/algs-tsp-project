import numpy as np
from save_tsp_file import save_matrix_file
from helpers import add_sparsity

def generate_atsp(n, dim_size=100, sparsity=0):
    # Generate a random asymmetric distance matrix
    distance_matrix = np.random.randint(1, dim_size, size=(n, n))
    np.fill_diagonal(distance_matrix, 0)
    if sparsity > 0:
        distance_matrix = add_sparsity(distance_matrix, sparsity, symmetric=False)

    save_matrix_file(matrix=distance_matrix, filename="random_atsp.atsp",type="ATSP")

# Example usage
if __name__ == "__main__":
  generate_atsp(8, dim_size=100, sparsity=0.7)