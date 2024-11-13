# %%
# Traveling Salesman Problem
# Dynamic Programming Approach
# Developer: Chelsy Mena

#%% Library Imports

import tsplib95
import numpy as np
from itertools import combinations

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

#%% Execution

# load a problem
problem = tsplib95.load('ALL_tsp/ch130.tsp')
distance_matrix = create_distance_matrix(problem)

nodes = list(problem.get_nodes())
first_city = determine_start(problem)

#%%
# Getting the cost of coming back if you are the last one
distances = {}
paths = {}

for v in nodes:
	#path = [v, first_city]
	dist = calculate_distance(distance_matrix, v, first_city)

	distances[(frozenset([v]), v)] = dist

nodes.remove(first_city)

i = 2
for i in range(2, len(nodes)+1):

	# generar subsets de nodos y mirar si ya estan en la tabla o no, e ir haciendo el de la tabla + el del par que queda
	for S in combinations(nodes, i):

		subset = frozenset(S)

		for w in subset:
			distances[(subset, w)] = float(np.inf)

			for u in subset - {w}:
				dist = distances[(subset- {w}, u)] + calculate_distance(distance_matrix, u, w)

				if dist < distances[(subset, w)]: #camino hasta w
					distances[(subset, w)] = dist
					paths[(subset, w)] = u

# Assuming 'V' is the set of cities, and 'start_city' is your arbitrary start city (e.g., v)
full_set = frozenset(nodes)  # All cities visited, except starting city
min_cost = float('inf')  # Initialize with infinity

# Look for the minimum cost to complete the tour by considering all possible end cities 'w'
for w in nodes:
    if w != first_city:  # Skip the starting city
        # The cost of completing the tour through city 'w' and then returning to start_city
        cost = distances[(full_set, w)] + calculate_distance(distance_matrix, w, first_city)  # Add cost to return to the start city
        min_cost = min(min_cost, cost)

print("Optimal cost:", min_cost)
# %%
