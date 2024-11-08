import tsplib95
import numpy as np
from collections import defaultdict

# Function to create the distance matrix from the TSP problem
def create_distance_matrix(problem):
    nodes = list(problem.get_nodes())
    n = problem.dimension
    distance_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = problem.get_weight(nodes[i], nodes[j])
    return distance_matrix

# Function to find the minimum cost path for all the paths
def findMinRoute(tsp):
    n = len(tsp)
    sum = 0
    counter = 0
    j = 0
    i = 0
    min = float('inf')
    visitedRouteList = defaultdict(int)

    # Starting from the 0th indexed city i.e., the first city
    visitedRouteList[0] = 1
    route = [0] * (n + 1)  # Ensure the route list is large enough
    route[0] = 1  # Start from the first city

    # Traverse the adjacency matrix tsp[][]
    while i < n and j < n:

        # Corner of the Matrix
        if counter >= n - 1:
            break

        # If this path is unvisited then and if the cost is less then update the cost
        if j != i and (visitedRouteList[j] == 0):
            if tsp[i][j] < min:
                min = tsp[i][j]
                route[counter + 1] = j + 1

        j += 1

        # Check all paths from the ith indexed city
        if j == n:
            sum += min
            min = float('inf')
            visitedRouteList[route[counter + 1] - 1] = 1
            j = 0
            i = route[counter + 1] - 1
            counter += 1

    # Update the ending city in array from city which was last visited
    i = route[counter] - 1

    for j in range(n):
        if (i != j) and tsp[i][j] < min:
            min = tsp[i][j]
            route[counter + 1] = j + 1

    sum += min

    # Output the ordered sequence of nodes to visit and the cost
    print("Ordered sequence of nodes to visit:", route[:counter + 2])
    print("Minimum Cost is:", sum)

# Driver Code
if __name__ == "__main__":
    file_path = 'ALL_tsp/bays29.tsp'
    problem = tsplib95.load(file_path)
    distance_matrix = create_distance_matrix(problem)
    print("Distance Matrix:")
    print(distance_matrix)
    findMinRoute(distance_matrix)