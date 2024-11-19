import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

#%% Function Definitions

def create_distance_matrix(problem):
    
	"""
		Function to create the distance matrix from the TSP problem
		
		Input: TSP problem in any of the accepted formats for the tsplib
		
		Output: numpy matrix with the edge distances for the problem
    """

	nodes = list(problem.get_nodes())
	n = problem.dimension
	distance_matrix = np.zeros((n, n), dtype=int)
	for i in range(n):
		for j in range(n):
			if i != j:
				distance_matrix[i][j] = problem.get_weight(nodes[i], nodes[j])
	return distance_matrix


def determine_start(problem):

	"""
		Function to choose a random node to start the search in
		
		Input: TSP problem in any of the accepted formats fot the tsp lib
		
		Output: Node to start with as an index k, used in the distance matrix
	"""

	n = problem.dimension

	starting_node = np.random.randint(0, n+1)

	return starting_node


def calculate_distance(distance_matrix, start_node, end_node):

	"""
		Function to get the distance between two nodes

		Input: Distance matrix, start and end nodes

		Output: Distance between the two nodes
	"""

	distance = distance_matrix[start_node - 1][end_node - 1]

	return int(distance)


def plot_graph(matrix, title=""):

	# Create a graph object
	G = nx.Graph()

	# Add edges with weights from the matrix
	num_nodes = len(matrix)
	for i in range(num_nodes):
		for j in range(num_nodes):  # Avoid self-loops and duplicate edges
			if matrix[i, j] > 0:  # Skip zero values (no edge)
				G.add_edge(i + 1, j + 1, weight=matrix[i, j])

	# Draw the MST graph
	pos = nx.spring_layout(G, seed=42)  # Position nodes using spring layout
	plt.figure(figsize=(10, 8))

	# Draw nodes and edges
	nx.draw_networkx_nodes(G, pos, node_size=500, node_color="skyblue")
	nx.draw_networkx_edges(G, pos, width=2)
	nx.draw_networkx_labels(G, pos, font_size=12, font_color="black")

	# Draw edge labels with weights
	edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
	nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

	plt.title(title)
	plt.axis("off")
	plt.show()