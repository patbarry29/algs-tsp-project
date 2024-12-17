# %%
# Traveling Salesman Problem
# Randomized Approach - Markov Chain Monte Carlo Approximation
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

def randomized(distance_matrix):

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
	T = 1

	# Set plateau counter
	plateau_limit = 5000
	plateau_counter = 0

	results = []
	for i in range(10001):

		for j in range(n_nodes*5):

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
			   (T > 0.00000005 and np.random.rand(1)[0] < np.exp(-delta/T)):
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
		if (i > 0) and (i % 500 == 0):
			T *= 1.2
			print(f"I'm in iteration {i}, don't kill me. I have kids.")

	return route, dist_best

#%% Execution Example
import matplotlib.pyplot as plt
from gui.opt_solutions import opt_solutions

problems = list(opt_solutions.keys())

opt_solutions_routes = {}

to_plot = [["problem", "opt_cost", "calc_cost"]]

for problem_name in problems:
	if ".atsp" in problem_name:
		folder = r"C:\Users\USER\iCloudDrive\iCloud~md~obsidian\iCloud Vault\Masters\Advanced Algorithmics and Programming\Project\algs-tsp-project\data\ALL_atsp\\"
	else:
		folder = folder = r"C:\Users\USER\iCloudDrive\iCloud~md~obsidian\iCloud Vault\Masters\Advanced Algorithmics and Programming\Project\algs-tsp-project\data\ALL_tsp\\"

	problem = tsplib95.load(f"{folder}{problem_name}")
	distance_matrix = create_distance_matrix(problem)

	i = 0
	while i < 50:
		results = randomized(distance_matrix)

		to_plot.append([
			problem_name,
			opt_solutions[problem_name],
			results[1]
		])

		if opt_solutions[problem_name] == results[1]:
			opt_solutions_routes[problem_name] = (opt_solutions[problem_name], [int(x) for x in results[0]])

		i += 1

	np.savetxt('randomized_cities_results_3.txt', np.array(to_plot), delimiter=' ', newline='\n', fmt="%s")

	with open(r"C:\Users\USER\iCloudDrive\iCloud~md~obsidian\iCloud Vault\Masters\Advanced Algorithmics and Programming\Project\algs-tsp-project\randomized\paths_3.txt", "w") as file:
			for key, value in opt_solutions_routes.items():
				file.write(f"{key}: {value}\n")

# #%%
# plt.plot(range(len(results)), results)
# plt.title(f'p size = {len(distance_matrix)}, Cost = {results[1]}')
# plt.ylabel("Cost")
# plt.xlabel("Iterations");
# %%

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

df = pd.DataFrame(to_plot[1:], columns = to_plot[0])
df['n_cities'] = df['problem'].apply(lambda x: int(''.join([char for char in x if char.isdigit()])))
df['deviation'] = ((df['calc_cost'] - df['opt_cost'])/df['opt_cost'])*100
data_plots = df.groupby('problem').agg(['mean', 'std'])
df['type'] =  df['problem'].apply(lambda x: x.partition('.')[2])
data_plots['type'] = [x.partition('.')[2] for x in data_plots.index]

# Setup 4-plot layout
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
#axes = axes.flatten()

sns.scatterplot(
	data=data_plots,
	x=data_plots[('n_cities','mean')], 
	y=data_plots[('deviation', 'mean')],
	hue=data_plots['type'], ax=axes[0])
axes[0].set_title("Average Deviation vs Number or Cities")
axes[0].set_ylabel("Deviation")
axes[0].set_xlabel("Number of Cities")

problems_to_show = ['burma14.tsp', 'st70.tsp', 'ch130.tsp']
# KDE plots
for i in range(1, 4):
	temp_df = df[df['problem'] == problems_to_show[i-1]]
	sns.kdeplot(temp_df['calc_cost'], ax=axes[i])

	opt = np.mean(temp_df['opt_cost'])
	calc = np.mean(temp_df['calc_cost']) # Example of additional vertical line

	# Highlight mean - Green
	axes[i].axvline(opt, color='#1cd12b', linestyle='--', label=f'Opt Cost = {opt:.1f}')
	axes[i].annotate(f'Opt Cost = {opt:.1f}', 
						xy=(opt, 0.1), 
						xytext=(opt + 0.5, 0.15), 
						arrowprops=dict(arrowstyle='->', color='#1cd12b'), color='#1cd12b')

	# Additional vertical line
	axes[i].axvline(calc, color='#d1911c', linestyle='--', label=f' Avg Calc Cost = {calc:.1f}')
	axes[i].annotate(f'Avg Calc Cost = {calc:.1f}', 
						xy=(calc, 0.1), 
						xytext=(calc + 0.5, 0.15), 
						arrowprops=dict(arrowstyle='->', color='#d1911c'), color='#d1911c')

	axes[i].set_title(f"Calculated Best Cost Distribution for {problems_to_show[i-1].partition('.')[0]}")
	axes[i].legend()
	axes[i].set_xlabel("")
	axes[i].set_ylabel("")

plt.tight_layout()
plt.show()
# %%
