# %%
# Traveling Salesman Problem
# Dynamic Programming Approach
# Developer: Chelsy Mena

#%% LIBRARY IMPORTS
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

import numpy as np
from itertools import combinations

#%% FUNCTION DEFINITIONS

def determine_start(nodes):

	"""
		Function to choose a random node to start the search in

		Input: TSP problem in any of the accepted formats fot the tsp lib
		Output: Node to start with as an index k, used in the distance matrix
	"""

	n = len(nodes)

	starting_node = np.random.randint(1, n+1)

	return starting_node


def calculate_distance(distance_matrix, start_node, end_node):

	"""
		Function to get the distance between two nodes

		Input: Distance matrix, start and end nodes
		Output: Distance between the two nodes
	"""

	distance = distance_matrix[start_node - 1][end_node - 1]

	return float(distance)


def dynamic_programming(distance_matrix):

	"""
		Obtains a path and the cost of it for TSP

		Inputs
			- np.array with the distance matrix

		Output:
			- Tuple where the first element is the best path, a python list
			- The second element is the cost of the path, a float
	"""
	nodes = list(np.arange(distance_matrix.shape[0])+1)
	nodes = [int(x) for x in nodes]
	first_city = determine_start(nodes)

	# Getting the cost of coming back if you are the last one
	distances = {}
	paths = {}

	for v in nodes:
		dist = calculate_distance(distance_matrix, v, first_city)
		distances[(frozenset([v]), v)] = dist

	print(first_city, nodes)
	nodes.remove(first_city)

	# Calculating the costs of all possible paths, and storing the mins
	for i in range(2, len(nodes)+1):

		# Making nodes subsets
		for S in combinations(nodes, i):

			subset = frozenset(S)

			for w in subset:
				distances[(subset, w)] = float(np.inf)

				for u in subset - {w}:
					dist = distances[(subset-{w}, u)] + calculate_distance(distance_matrix, u, w)

					if dist < distances[(subset, w)]:
						distances[(subset, w)] = dist # Rewriting the best cost from the subset to the current city
						paths[(subset, w)] = u	  # Writing the best last city from this subset to the current city

	# Getting the smallest cost
	full_set = frozenset(nodes)  # All cities visited, except starting city
	min_cost = float(np.inf)

	# Look for the minimum distance to complete the tour
	for w in nodes:
		cost = distances[(full_set, w)] + calculate_distance(distance_matrix, w, first_city)  # Add cost to return to the start city
		if cost < min_cost:
			min_cost = min(min_cost, cost)
			final_city = w

	# Getting the path
	path = []
	current_city = final_city
	current_subset = full_set

	while (len(path) < len(nodes)-1):
		path.append(current_city)
		previous_city = paths[(current_subset, current_city)]  # Get the predecessor
		current_subset = current_subset - {current_city}
		current_city = previous_city

	# Add the final city and the start city to close the tour
	path.append(current_city)
	path.append(first_city)
	path.reverse()

	return (path, min_cost)