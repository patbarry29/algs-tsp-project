import sys
import os
import time  # To measure the time taken
from queue import PriorityQueue
import tsplib95  # For loading the TSP problem
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from utils.create_distance_matrix import create_distance_matrix  # Import the distance matrix function

# Define the number of cities (N) and the infinity value (INF)
INF = sys.maxsize

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
        
        # Change the row i and column j to INF to avoid revisiting
        if level != 0:
            for k in range(N):
                self.reducedMatrix[i][k] = INF
                self.reducedMatrix[k][j] = INF
            self.reducedMatrix[j][0] = INF  # Don't return to the start city (0)

    def __lt__(self, other):
        return self.cost < other.cost

# Function to perform row reduction
def rowReduction(reducedMatrix):
    row = [INF] * N
    for i in range(N):
        for j in range(N):
            if reducedMatrix[i][j] < row[i]:
                row[i] = reducedMatrix[i][j]
    for i in range(N):
        for j in range(N):
            if reducedMatrix[i][j] != INF and row[i] != INF:
                reducedMatrix[i][j] -= row[i]
    return row

# Function to perform column reduction
def columnReduction(reducedMatrix):
    col = [INF] * N
    for i in range(N):
        for j in range(N):
            if reducedMatrix[i][j] < col[j]:
                col[j] = reducedMatrix[i][j]
    for i in range(N):
        for j in range(N):
            if reducedMatrix[i][j] != INF and col[j] != INF:
                reducedMatrix[i][j] -= col[j]
    return col

# Function to calculate the cost of the path
def calculateCost(reducedMatrix):
    cost = 0
    row = rowReduction(reducedMatrix)
    col = columnReduction(reducedMatrix)

    # Sum the row and column reduction values
    for i in range(N):
        cost += (row[i] if row[i] != INF else 0)
        cost += (col[i] if col[i] != INF else 0)

    return cost

# Function to print the path (tour)
def printPath(path):
    print("Path taken:")
    for pair in path:
        print(f"{pair[0] + 1} -> {pair[1] + 1}")

# Function to solve the TSP problem using Branch and Bound
def solve(CostGraphMatrix):
    pq = PriorityQueue()

    # Create a root node and calculate its cost
    root = Node(CostGraphMatrix, [], 0, -1, 0)
    root.cost = calculateCost(root.reducedMatrix)

    # Add root to the list of live nodes
    pq.put((root.cost, root))

    # Continue until the priority queue becomes empty
    while not pq.empty():
        min = pq.get()[1]  # Get node with minimum cost

        i = min.vertex

        # If all the cities have been visited, complete the tour
        if min.level == N - 1:
            min.path.append((i, 0))  # Return to the start city
            printPath(min.path)  # Print the path taken
            return min.cost

        # Generate all the children of the current node
        for j in range(N):
            if min.reducedMatrix[i][j] != INF:
                child = Node(min.reducedMatrix, min.path, min.level + 1, i, j)
                edge_cost = min.reducedMatrix[i][j]
                child.cost = min.cost + edge_cost + calculateCost(child.reducedMatrix)

                # Print edge selection and cost
                print(f"Edge selected: {i + 1} -> {j + 1}, Cost: {edge_cost}")

                pq.put((child.cost, child))

    return 0

# Example usage with a TSPLIB problem
if __name__ == "__main__":
    # Load the TSP problem from a file using tsplib95
    problem = tsplib95.load('data/random/tsp/random_tsp.tsp')  # Load the problem using tsplib95
    distance_matrix_data = create_distance_matrix(problem)  # Use the imported function to get the matrix
    N = len(distance_matrix_data)

    # Print the initial distance matrix
    print("Initial Distance Matrix:")
    print(distance_matrix_data)

    # Measure the time taken for the TSP solver
    start_time = time.time()  # Start timer
    total_cost = solve(distance_matrix_data)
    end_time = time.time()  # End timer

    # Print the total cost of the tour
    print(f"Total cost is: {total_cost}")

    # Print the time taken
    print(f"Time taken: {end_time - start_time:.4f} seconds")
