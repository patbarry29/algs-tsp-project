# %%
# Traveling Salesman Problem
# Randomised Approach - Not really, was going to be Cristofides
# Developer: Chelsy Mena

#%% LIBRARY IMPORTS

import tsplib95
import numpy as np
from itertools import product, combinations
from utils import create_distance_matrix, determine_start, calculate_distance, plot_graph

from tsp_problems import opt_solution

#%% Load a problem
file = 'burma14'
problem = tsplib95.load(f'ALL_tsp/{file}.tsp')
distance_matrix = create_distance_matrix(problem)

#%% MINIMUM SPANNING TREE - compared to mst with nx.minimum_spanning_tree(G), looks ok

nodes = list(problem.get_nodes())
first_city = determine_start(problem)

mst_nodes = [first_city]
mst_matrix = np.zeros((len(nodes), len(nodes)))

nodes.remove(first_city)

for i in range(len(nodes)):
	dists_mst = []
	nodes_tried = []

	for chosen, trying in product(mst_nodes, nodes):
		# Try every node
		distance = calculate_distance(
			distance_matrix, 
			chosen, trying)
		dists_mst.append(distance)
		nodes_tried.append((chosen, trying))

	# Add the one closest to the MST seen nodes
	index_min = dists_mst.index(min(dists_mst))
	pair_nodes = nodes_tried[index_min]
	mst_nodes.append(pair_nodes[1])

	# Put the distance in the matrix
	mst_matrix[pair_nodes[0] - 1, pair_nodes[1] - 1] = min(dists_mst)
	mst_matrix[pair_nodes[1] - 1, pair_nodes[0] - 1] = min(dists_mst)

	# Remove the new node from the unseen set
	nodes.remove(pair_nodes[1])

plot_graph(mst_matrix)
#%% PERFECT MATCHING FOR THE NODES WITH ODD EDGES

# %% ADDING UP THE TWO GRAPHS
		
