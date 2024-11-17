import numpy as np
from save_tsp_file import save_matrix_file
from utils.add_sparsity import add_sparsity
from utils.random_edge_weights import generate_random_weights, generate_weight_distribution

def generate_tsp(n, dim_size=100, sparsity=0, random_weights=False, weight_dist=False):
    cities = np.random.uniform(0, dim_size, size=(n, 2))
    distance_matrix = None
    if sparsity > 0:
        distance_matrix = np.array([[np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2) for x1 in cities] for x2 in cities])
        distance_matrix = add_sparsity(distance_matrix, sparsity, symmetric=True)
    save_matrix_file(cities, distance_matrix)
    return cities

if __name__ == "__main__":
  	cities = generate_tsp(n=5, dim_size=10, sparsity=0.8)