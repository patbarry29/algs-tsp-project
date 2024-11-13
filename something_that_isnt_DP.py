# %%
# Traveling Salesman Problem
# Dynamic Programming Approach
# Developer: Chelsy Mena

#%% Library Imports

import tsplib95
import numpy as np

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

	return distance

#%% Execution

# load a problem
problem = tsplib95.load('ALL_tsp/att48.tsp')
distance_matrix = create_distance_matrix(problem)

# Get all distances from the starting node
nodes_ext = list(problem.get_nodes())
start_node = determine_start(problem)

# Getting the cost of coming back from the last one
# aka just the line of the matrix for the starting node
distances_return = []

for v in nodes_ext:
	if v == start_node: 
		distances_return.append(np.inf)
	if v != start_node:
		distance_return = calculate_distance(
			distance_matrix, start_node, end_node=v
			)
		distances_return.append(distance_return)

#paths = []
costs = []
nodes_ext = list(problem.get_nodes())
nodes_ext.remove(start_node)

nodes = list(problem.get_nodes())
nodes.remove(start_node)

print("Starting node:", start_node)

for w in nodes_ext:

	print("Trying first layer node: ", w)
	total_cost_path_chosen = distances_return[w-1]
	
	nodes.remove(w)
	current_path_chosen = [w, np.inf]

	while len(nodes) > 1:
		distance_current_path = np.inf

		for u in nodes:
			print("Trying second layer node: ", u)

			distance = total_cost_path_chosen + calculate_distance(
				distance_matrix, 
				current_path_chosen[-2], 
				u)
			
			if distance < distance_current_path:
				distance_current_path = distance
				current_path_chosen[-1] = u
		
		total_cost_path_chosen += distance_current_path
		print("CHOSE second layer node: ", current_path_chosen[-1])

		try:
			nodes.remove(current_path_chosen[-1])
			print("remaining possible nodes: ", len(nodes))
		except:
			print("remaining possible nodes: ", len(nodes))
			continue

		current_path_chosen.append(np.inf)

	costs.append([
		w,
		int(total_cost_path_chosen),
		str(start_node) + ', ' + ', '.join([str(x) for x in current_path_chosen[:-1]])
		])

	nodes = list(problem.get_nodes())
	nodes.remove(start_node)

print(np.array(costs))

#%%
# iterations = []
# for i in range(1, problem.dimension): #i es el tamaño del subset de nodos a revisar

# 	for w in nodes:

# 		if (w != start_node):

# 			index_w = paths.index(w)
# 			ith_sized_distance = distances[index_w]
# 			ith_sized_path = start_node

# 			line = [i, w]

# 			for u in nodes:
				
# 				if (u != w) and (u != start_node):

# 					distance = ith_sized_distance +\
# 						  calculate_distance(distance_matrix, w, u)

# 					if distance < ith_sized_distance:
# 						ith_sized_distance = distance
# 						ith_sized_path = str(u)

# 						line.append((ith_sized_distance, u))

# 					print(f"""
# Size of problem: {i}
# Final distance: {ith_sized_distance}
# Final path: {ith_sized_path}
# 						""")
# 			iterations.append(line)

# that's just a row of the matrix 

# store the paths in a table P


# %%