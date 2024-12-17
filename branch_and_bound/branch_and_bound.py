import heapq  
import numpy as np
import tsplib95
import os
import sys
import time  
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(utils_dir)

from utils.create_distance_matrix import create_distance_matrix  # Import distance matrix utility


def calculate_bound(path, distance_matrix):
    
    bound = 0
    num_nodes = len(distance_matrix)
    visited = set(path)

  
    for i in range(len(path) - 1):
        bound += distance_matrix[path[i]][path[i + 1]]

   
    for i in range(num_nodes):
        if i not in visited:
            min_edge = min(distance_matrix[i][j] for j in range(num_nodes) if i != j)
            bound += min_edge

    return bound


def generate_children(path, num_nodes):
    
    children = []
    for i in range(num_nodes):
        if i not in path:  # If not visited, add as a child path
            children.append(path + [i])
    return children


def branch_and_bound(distance_matrix):
    
    num_nodes = len(distance_matrix)
    best_tour = None
    best_cost = float('inf')

    
    pq = []
    initial_path = [0]  # Start from node 0
    initial_bound = calculate_bound(initial_path, distance_matrix)
    heapq.heappush(pq, (initial_bound, 0, initial_path))

    while pq:
        current_bound, current_cost, current_path = heapq.heappop(pq)

       
        if current_bound >= best_cost:
            continue

        
        if len(current_path) == num_nodes:
            final_cost = current_cost + distance_matrix[current_path[-1]][current_path[0]]
            if final_cost < best_cost:
                best_cost = final_cost
                best_tour = current_path + [0]  # Return to starting node
            continue

       
        for child in generate_children(current_path, num_nodes):
            last_city = current_path[-1]
            next_city = child[-1]
            new_cost = current_cost + distance_matrix[last_city][next_city]
            new_bound = calculate_bound(child, distance_matrix)

           
            if new_bound < best_cost:
                heapq.heappush(pq, (new_bound, new_cost, child))

    return best_tour, best_cost


if __name__ == "__main__":
    # Load a TSP problem using tsplib95
    problem = tsplib95.load('data/random/tsp/random_tsp.tsp')

    # Create the distance matrix from the TSP problem
    distance_matrix = create_distance_matrix(problem)

    # Measure the start time
    start_time = time.time()

    # Solve the TSP using Branch and Bound
    print("Solving TSP with Branch and Bound...\n")
    best_tour, best_cost = branch_and_bound(distance_matrix)

    # Measure the end time
    end_time = time.time()
    time_taken = end_time - start_time

    # Display the results
    print(f"Best Tour: {best_tour}")
    print(f"Best Cost: {best_cost}")
    print(f"Time Taken: {time_taken:.4f} seconds")
