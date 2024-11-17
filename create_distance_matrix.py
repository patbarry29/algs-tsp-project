import numpy as np

# Function to create the distance matrix from the TSP problem
def create_distance_matrix(problem):
    nodes = list(problem.get_nodes())
    num_nodes = problem.dimension
    distance_matrix = np.zeros((num_nodes, num_nodes), dtype=float)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                distance_matrix[i][j] = problem.get_weight(nodes[i], nodes[j])
    print(distance_matrix)
    return distance_matrix