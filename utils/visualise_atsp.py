import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

import tsplib95
from utils.create_distance_matrix import create_distance_matrix
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def visualize_atsp(distance_matrix, tsp_algorithm):
    """
    Visualizes an ATSP graph using NetworkX's DiGraph (directed graph) to show edge directions.
    Highlights the TSP solution path with red arrows and black nodes.

    Parameters:
    - distance_matrix: 2D numpy array representing the distances between nodes.
    - tsp_algorithm: Function that returns a tuple (cost, sequence) where sequence is the path of nodes.
    """
    # Create a directed graph
    G = nx.DiGraph()

    # Get the number of nodes
    num_nodes = distance_matrix.shape[0]

    # Add nodes to the graph
    G.add_nodes_from(range(num_nodes))

    # Add edges with weights (only for non-zero distances)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:  # No self-loops
                weight = distance_matrix[i][j]
                G.add_edge(i, j, weight=weight)

    # Use NetworkX's layout function (spring layout)
    pos = nx.spring_layout(G, seed=42)

    # Draw the nodes
    plt.figure(figsize=(10, 7))
    nx.draw_networkx_nodes(G, pos, node_color='black', node_size=50)

    # Draw the directed edges with arrowheads
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', arrowstyle='-|>', arrowsize=20, width=0.5)

    # Run the TSP algorithm to get the TSP path and cost
    sequence, cost = tsp_algorithm(distance_matrix)

    # Highlight the TSP path (convert 1-indexed to 0-indexed)
    for i in range(len(sequence) - 1):
        u, v = sequence[i] - 1, sequence[i + 1] - 1  # Convert to 0-indexed
        if u >= 0 and v >= 0 and u < num_nodes and v < num_nodes:
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], edge_color='red', width=2, arrowstyle='-|>', arrowsize=10
            )

    # Connect the last node back to the first to complete the cycle
    u, v = sequence[-1] - 1, sequence[0] - 1  # Convert to 0-indexed
    if u >= 0 and v >= 0 and u < num_nodes and v < num_nodes:
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], edge_color='red', width=2, arrowstyle='-|>', arrowsize=10
        )

    # Display the plot
    plt.title(f"ATSP Visualization with TSP Path (Cost: {cost})")
    plt.axis("off")  # Turn off the axis
    plt.show()


if __name__ == "__main__":
    # Load the ATSP problem
    problem = tsplib95.load(f'../data/ALL_atsp/ftv47.atsp')

    # Create the distance matrix
    distance_matrix = create_distance_matrix(problem)

    # Visualize the ATSP graph
    #visualize_atsp(distance_matrix, greedy)
