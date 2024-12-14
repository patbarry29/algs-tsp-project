import sys
import os
import numpy as np
import time  # To measure the time taken
from queue import PriorityQueue
import tsplib95  # For loading the TSP problem
import matplotlib.pyplot as plt  # Importing for plotting

# Add parent directory for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from utils.create_distance_matrix import create_distance_matrix  

# Node class to store each node along with the cost, level, and vertex
class Node:
    def __init__(self, parentMatrix, path, level, i, j):
        self.path = path.copy()  # Copy the path from the parent node
        self.reducedMatrix = [row.copy() for row in parentMatrix]  # Copy the reduced matrix
        self.cost = 0  # The cost of this node
        self.vertex = j  # Current city vertex
        self.level = level  # Level (or depth) in the search tree

        # Add this edge to the path
        if level != 0:
            self.path.append((i, j))

        # Change the row i and column j to infinity to avoid revisiting
        if level != 0:
            for k in range(N):
                self.reducedMatrix[i][k] = float('inf')
                self.reducedMatrix[k][j] = float('inf')
            self.reducedMatrix[j][0] = float('inf')  # Don't return to the start city (0)

    def __lt__(self, other):
        return self.cost < other.cost


# Perform row reduction
def rowReduction(reducedMatrix):
    row = [float('inf')] * N
    for i in range(N):
        for j in range(N):
            if reducedMatrix[i][j] < row[i]:
                row[i] = reducedMatrix[i][j]
    for i in range(N):
        for j in range(N):
            if reducedMatrix[i][j] != float('inf') and row[i] != float('inf'):
                reducedMatrix[i][j] -= row[i]
    return sum([r for r in row if r != float('inf')])


# Perform column reduction
def columnReduction(reducedMatrix):
    col = [float('inf')] * N
    for i in range(N):
        for j in range(N):
            if reducedMatrix[i][j] < col[j]:
                col[j] = reducedMatrix[i][j]
    for i in range(N):
        for j in range(N):
            if reducedMatrix[i][j] != float('inf') and col[j] != float('inf'):
                reducedMatrix[i][j] -= col[j]
    return sum([c for c in col if c != float('inf')])


# Calculate the cost of the path using row and column reductions
def calculateCost(reducedMatrix):
    cost = 0
    cost += rowReduction(reducedMatrix)
    cost += columnReduction(reducedMatrix)
    return cost

# Function to handle edge selection and log edge details
def edge_selection(parent_node, i, j):
    edge_cost = parent_node.reducedMatrix[i][j]
    print(f"Edge selected: {i + 1} -> {j + 1}, Cost: {edge_cost}")
    return edge_cost

# Solve TSP using Branch and Bound with tracking for convergence
def branch_and_bound(CostGraphMatrix):
    pq = PriorityQueue()
    convergence_costs = []  # List to track costs at each iteration

    # Perform initial reduction and create root node
    root = Node(CostGraphMatrix, [], 0, -1, 0)
    root.cost = calculateCost(root.reducedMatrix)
    pq.put((root.cost, root))

    # While there are live nodes
    while not pq.empty():
        current_node = pq.get()[1]  # Get the node with the lowest cost
        convergence_costs.append(current_node.cost)  # Track current cost

        i = current_node.vertex

        # If all cities are visited, complete the tour
        if current_node.level == N - 1:
            current_node.path.append((i, 0))  # Append the return to the start
            # Convert path to 1-based indexing for readability
            converted_path = [u + 1 for u, v in current_node.path]
            return current_node.cost, converted_path, convergence_costs  # Return both cost and 1-based path

        # Generate child nodes for the current node
        for j in range(N):
            if current_node.reducedMatrix[i][j] != float('inf'):
                child = Node(current_node.reducedMatrix, current_node.path, current_node.level + 1, i, j)
                edge_cost = current_node.reducedMatrix[i][j]
                child.cost = current_node.cost + current_node.reducedMatrix[i][j]
                child.cost += calculateCost(child.reducedMatrix)
                
                pq.put((child.cost, child))

    return float('inf'), [], convergence_costs  # Return a default path when no solution is found


# Calculate the final cost based on the original matrix
def calculateFinalCost(path, originalMatrix):
    total_cost = 0
    for i in range(len(path) - 1):
        total_cost += originalMatrix[path[i] - 1][path[i + 1] - 1]  # Convert back to 0-based indexing for matrix lookup
    return total_cost, path


# Plot convergence
def plot_convergence(costs):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(costs)), costs, marker='o', linestyle='-', color='b')
    plt.title('Convergence of Branch and Bound')
    plt.xlabel('Iteration')
    plt.ylabel('Total Cost')
    plt.grid()
    # Print total cost on the plot
    plt.text(len(costs) - 1, costs[-1], f'Total Cost = {costs[-1]}', 
             horizontalalignment='right', verticalalignment='bottom', fontsize=12, color='red')
    plt.show()


# Example usage
if __name__ == "__main__":
    # Load the TSP problem from a file using tsplib95
    problem = tsplib95.load('data/random/atsp/random_atsp.atsp')  # Load the problem using tsplib95
    distance_matrix_data = create_distance_matrix(problem)  # Use the imported function to get the matrix
    N = len(distance_matrix_data)

    # Set diagonal to infinity to prevent self-loops
    np.fill_diagonal(distance_matrix_data, float('inf'))

    # Print the initial distance matrix
    print("Initial Distance Matrix:")
    print(distance_matrix_data)

    # Measure the time taken to solve the problem
    start_time = time.time()
    total_cost, final_path, convergence_costs = branch_and_bound(distance_matrix_data)
    end_time = time.time()

    # Recalculate the cost using the original matrix
    final_cost, adjusted_path = calculateFinalCost(final_path, distance_matrix_data)

    # Print the results
    print(f"\nTotal cost based on reduced matrix: {total_cost}")
    print("Final Path (1-based indexing):", adjusted_path)
    print(f"Time taken: {end_time - start_time:.4f} seconds")

    # Plot the convergence
    plot_convergence(convergence_costs)
