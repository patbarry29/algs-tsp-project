import numpy as np

def add_sparsity(distance_matrix, sparsity, symmetric):
    num_nodes = distance_matrix.shape[0]

    # 1. Create a random path that starts and ends at 0
    path = np.random.permutation(num_nodes-1) + 1
    path = np.append(path, 0)
    path = np.insert(path, 0, 0)

    # 2. Initialize mask with all False (no edges)
    mask = np.zeros((num_nodes, num_nodes), dtype=bool)

    # 3. Add edges along the random path
    for i in range(num_nodes - 1):
        mask[path[i], path[i+1]] = True
        if symmetric:
          mask[path[i+1], path[i]] = True

    mask[path[-1], path[0]] = True
    if symmetric:
      mask[path[0], path[-1]] = True

    # 4. Add random edges based on sparsity, excluding the random path edges
    random_mask = np.random.rand(num_nodes, num_nodes) > sparsity
    mask = np.logical_or(mask, np.logical_and(random_mask, np.logical_not(mask)))

    np.fill_diagonal(mask, False)

    # 6. Apply the mask to the distance matrix
    sparse_matrix = np.where(mask, distance_matrix, np.max(distance_matrix) ** 3)
    return sparse_matrix