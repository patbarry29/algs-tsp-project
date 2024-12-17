import os
import sys
import time
import tsplib95

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.create_distance_matrix import create_distance_matrix

# Adapted [Prim's MST implementation] from :
# https://www.geeksforgeeks.org/prims-minimum-spanning-tree-mst-greedy-algo-5/

def get_mst_edges(matrix):
    n = len(matrix)
    key = [float('inf')] * n
    parent = [None] * n 
    mst_set = [False] * n

    key[0] = 0
    parent[0] = 0

    for _ in range(n-1):
        min_key = float('inf')
        u = 1  
        for v in range(1, n+1): 
            if not mst_set[v-1] and key[v-1] < min_key:
                min_key = key[v-1]
                u = v

        mst_set[u-1] = True

        for v in range(1, n+1):
            if (matrix[u-1][v-1] > 0 and not mst_set[v-1] and 
                matrix[u-1][v-1] < key[v-1]):
                key[v-1] = matrix[u-1][v-1]
                parent[v-1] = u

    edges = []
    for i in range(1, n):
        edges.append((parent[i], i+1)) 
    return edges

def make_tour_from_mst(edges, n):
    adj = [[] for _ in range(n+1)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    visited = [False] * (n+1) 
    tour = []
    
    def dfs(v):
        visited[v] = True
        tour.append(v)
        for u in adj[v]:
            if not visited[u]:
                dfs(u)
                
    dfs(1) 
    tour.append(1)
    return tour

def route_distance(tour, distance_matrix):
    cost = 0
    for i in range(len(tour)-1):
        from_vertex = tour[i] - 1
        to_vertex = tour[i+1] - 1
        cost += distance_matrix[from_vertex][to_vertex]
    return cost

def mst(distance_matrix):
    n = len(distance_matrix)
    edges = get_mst_edges(distance_matrix)
    #print(edges)
    tour = make_tour_from_mst(edges, n)
    cost = route_distance(tour, distance_matrix)
    return tour, cost

# if __name__ == "__main__":
#     problem = tsplib95.load('data/ALL_tsp/burma14.tsp')
#     matrix = create_distance_matrix(problem)
    
#     start_time = time.time()
#     tour, cost = mst(matrix)
#     end_time = time.time() - start_time
    
#     print("Tour:", tour)
#     print("Cost:", cost) 
#     print(f"Time: {end_time:.6f}s")