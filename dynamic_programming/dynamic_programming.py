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

import tsplib95
import numpy as np
from itertools import combinations
from utils.create_distance_matrix import create_distance_matrix


#%% FUNCTION DEFINITIONS

def determine_start(nodes):

	"""
		Function to choose a random node to start the search in
		
		Input: TSP problem in any of the accepted formats fot the tsp lib
		Output: Node to start with as an index k, used in the distance matrix
	"""

	n = len(nodes)

	starting_node = np.random.randint(0, n+1)

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

	import os
	import psutil
	# Get the current process
	process = psutil.Process(os.getpid())

	nodes = list(np.arange(distance_matrix.shape[0])+1)
	nodes = [int(x) for x in nodes]
	first_city = determine_start(nodes)

	# Getting the cost of coming back if you are the last one
	distances = {}
	paths = {}

	for v in nodes:
		dist = calculate_distance(distance_matrix, v, first_city)
		distances[(frozenset([v]), v)] = dist

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

	memory_usage = process.memory_info().rss / (1024 * 1024)

	return (path, min_cost), memory_usage
#%% EXECUTION

from gui.opt_solutions import opt_solutions
import time

problems = list(opt_solutions.keys())[:8]

opt_solutions_routes = {}

to_plot = [["problem", "opt_cost", "calc_cost", 'time', 'memory']]

for problem_name in problems:
	if ".atsp" in problem_name:
		folder = r"C:\Users\USER\iCloudDrive\iCloud~md~obsidian\iCloud Vault\Masters\Advanced Algorithmics and Programming\Project\algs-tsp-project\data\ALL_atsp\\"
	else:
		folder = folder = r"C:\Users\USER\iCloudDrive\iCloud~md~obsidian\iCloud Vault\Masters\Advanced Algorithmics and Programming\Project\algs-tsp-project\data\ALL_tsp\\"

	problem = tsplib95.load(f"{folder}{problem_name}")
	distance_matrix = create_distance_matrix(problem)

	start_time = time.time()
	results, memory = dynamic_programming(distance_matrix)
	end_time = time.time()
	run_time = end_time - start_time

	to_plot.append([
		problem_name,
		opt_solutions[problem_name],
		results[1],
		run_time,
		memory
	])

	if opt_solutions[problem_name] == results[1]:
		opt_solutions_routes[problem_name] = (opt_solutions[problem_name], [int(x) for x in results[0]])

np.savetxt('dp_cities_results.txt', np.array(to_plot), delimiter=' ', newline='\n', fmt="%s")

with open(r"C:\Users\USER\iCloudDrive\iCloud~md~obsidian\iCloud Vault\Masters\Advanced Algorithmics and Programming\Project\algs-tsp-project\dynamic_programming\paths.txt", "w") as file:
		for key, value in opt_solutions_routes.items():
			file.write(f"{key}: {value}\n")
#%%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

df = pd.DataFrame(to_plot[5:], columns = to_plot[:5])
df['n_cities'] = df['problem'].apply(lambda x: int(''.join([char for char in x if char.isdigit()])))
df['deviation'] = ((df['calc_cost'] - df['opt_cost'])/df['opt_cost'])*100
#data_plots = df.groupby('problem').agg(['mean', 'std'])
df['type'] =  df['problem'].apply(lambda x: x.partition('.')[2])
#data_plots['type'] = [x.partition('.')[2] for x in data_plots.index]

# Setup 4-plot layout
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Horizontal bar chart
# sns.barplot(y=df['problem'], x=df['opt_cost'], ax=axs[0], label = "Optimal")
# sns.barplot(y=df['problem'], x=df['calc_cost'], ax=axs[0], label="Calculated")
# axs[0].set_title("Deviation from the known TSP/ATSP problems")
# axs[0].set_ylabel("Deviation")
# axs[0].set_ylabel("Problem")

# First lineplot
df_tsp = df[df['type']=="tsp"]
df_atsp = df[df['type']=="atsp"]
sns.lineplot(x=df_tsp['n_cities'], y=df_tsp['time'], ax=axs[1], label='tsp')
sns.scatterplot(x=df_atsp['n_cities'], y=df_atsp['time'], ax=axs[1], label='atsp')
axs[1].set_xticks(df_tsp['n_cities'])
axs[1].set_xticklabels(df_tsp['problem'], rotation=45)
axs[1].set_title("Execution Time vs Number of Cities")
axs[1].set_xlabel("Number of Cities")
axs[1].set_ylabel("Time (s)")

# Second lineplot
sns.lineplot(x=df_tsp['n_cities'], y=df_tsp['memory'], ax=axs[0], label="tsp")
sns.scatterplot(x=df_atsp['n_cities'], y=df_atsp['memory'], ax=axs[0], label='atsp')
axs[0].set_xticks(df_tsp['n_cities'])
axs[0].set_xticklabels(df_tsp['problem'], rotation=45)
axs[0].set_title("Memory Usage vs Number of Cities")
axs[0].set_xlabel("Number of Cities")
axs[0].set_ylabel("Memory Usage (MB)")

# Adjust layout
plt.tight_layout()
plt.show()

# %%