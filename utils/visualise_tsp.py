import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

import tsplib95
from utils.create_distance_matrix import create_distance_matrix
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import networkx as nx

def visualize_tsp(distance_matrix, tsp_algorithm, seed=42):
    """
    Visualize the data points and the TSP path.

    Parameters:
    - distance_matrix
    - tsp_algorithm
    - seed
    """
    G = nx.Graph()

    # Add nodes
    num_nodes = distance_matrix.shape[0]
    G.add_nodes_from(range(num_nodes))

    # Use MDS to compute 2D positions based on the distance matrix
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=seed)
    positions = mds.fit_transform(distance_matrix)
    pos = {i: positions[i] for i in range(num_nodes)}

    plt.axis("off")  # Remove axis borders
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove extra margins
    nx.draw_networkx_nodes(G, pos, node_color='black', node_size=10)  # Black nodes

    sequence, cost = tsp_algorithm(distance_matrix)

    # Plot the edges based on the TSP sequence
    for i in range(len(sequence) - 1):
        u, v = sequence[i] - 1, sequence[i + 1] - 1  # Convert to 0-indexed
        if u >= 0 and v >= 0 and u < num_nodes and v < num_nodes:
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color='blue', width=1)

    # Connect the last node back to the first to complete the cycle
    u, v = sequence[-1] - 1, sequence[0] - 1  # Convert to 0-indexed
    if u >= 0 and v >= 0 and u < num_nodes and v < num_nodes:
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color='blue', width=1)

    plt.show()


if __name__ == "__main__":
    problem = tsplib95.load(f'../data/ALL_tsp/d198.tsp')
    distance_matrix = create_distance_matrix(problem)
    #visualize_tsp(distance_matrix, greedy)
