# %%
# Traveling Salesman Problem
# Randomised Approach - Markov Chain Monte Carlo Approximation
# Developer: Chelsy Mena

#%% LIBRARY IMPORTS
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

import tsplib95
import numpy as np
from utils.create_distance_matrix import create_distance_matrix

# FUNCTION DEFINITIONS

def calculate_distance(distance_matrix, start_node, end_node):

	"""
		Function to get the distance between two nodes

		Input: Distance matrix, start and end nodes
		Output: Distance between the two nodes
	"""

	distance = distance_matrix[start_node - 1][end_node - 1]

	return int(distance)

def route_distance(distance_matrix, route):

	"""
		Gets the total distance cost of a given route

		Input: np.array, a distance matrix
		Output: float, the cost of the route with the closed loop
	"""

	dist_route = 0
	for i in range(1, len(route)):
		dist = calculate_distance(
			distance_matrix,
			route[i-1], route[i]
		)
		dist_route += dist
	dist_return = calculate_distance(
			distance_matrix,
			route[-1], route[0]
		)
	dist_route += dist_return

	return dist_route

def randomised(distance_matrix):

	"""
		Obtains a path and the cost of it for TSP

		Inputs
			- np.array with the distance matrix
		
		Output:
			- Tuple where the first element is the best path, a python list
			- The second element is the cost of the path, a float
			- List with the best cost in each iteration for performance evaluation
	"""

	n_nodes = len(distance_matrix)

	# Initialize the route randomly
	route = np.arange(1, n_nodes+1)
	np.random.shuffle(route)

	dist_route = route_distance(distance_matrix, route)
	dist_best = dist_route

	# Set Initial Temperature
	T = 100

	# Set plateau counter
	plateau_limit = 1000 
	plateau_counter = 0 

	results = []
	for i in range(10001):

		for j in range(200):

			# Flip path between Two random Cities
			current_route = route.copy()
			
			swap_index_1 = np.random.randint(n_nodes)
			swap_index_2 = np.random.randint(n_nodes)

			if swap_index_1 >= swap_index_2:
				swap_index_1, swap_index_2 = swap_index_2, swap_index_1 + 1
				current_route[swap_index_1:swap_index_2] = current_route[swap_index_1:swap_index_2][::-1]
			
			# Check if it's better
			delta = route_distance(distance_matrix, current_route) - dist_route

			# Accept if better, or randomly to jump to a new solution
			if (delta < 0) or \
				(T > 0.0000005 and np.random.rand(1)[0] < np.exp(-delta/T)):
				route = current_route.copy()
			
			dist_route = route_distance(distance_matrix, route)
			if  dist_route < dist_best:
				dist_best = dist_route
				plateau_counter = 0
			else:
				plateau_counter += 1
			
			results.append(dist_best)
		
		if plateau_counter >= plateau_limit:
			print(f"Converged after {i} iterations.")
			break
		
		T = 100*(0.9**i)

		# reheat every once in a while to avoid minimums
		if (i>0) and (i % 500 == 0):
			T *= 1.1
			print(f"I'm in iteration {i}, don't kill me. I have kids.")

	return (route, dist_best)
#%% Execution Example
# import matplotlib.pyplot as plt

# problem = tsplib95.load(r'data\ALL_tsp\burma14.tsp')
# distance_matrix = create_distance_matrix(problem)

# results_tuple, results = randomized(distance_matrix)

# plt.plot(range(len(results)), results)
# plt.title(f'p size = {len(distance_matrix)}, Cost = {results_tuple[1]}')
# plt.ylabel("Cost")
# plt.xlabel("Iterations");##