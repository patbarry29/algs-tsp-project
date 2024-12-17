import os
import sys
import time
import tsplib95
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.create_distance_matrix import create_distance_matrix

def route_distance(tour, distance_matrix):
    cost = 0
    for i in range(len(tour)-1):
        cost += distance_matrix[tour[i]][tour[i+1]]
    cost += distance_matrix[tour[-1]][tour[0]]
    return cost

def gain_crit(t1, t2, t3, t4, tsp_matrix):
    """Calculate gain for swapping edges"""
    return (tsp_matrix[t1][t2] + tsp_matrix[t3][t4] - 
            tsp_matrix[t1][t3] - tsp_matrix[t2][t4])

def create_edge(city1, city2):
    return (min(city1, city2), max(city1, city2))

def find_best_improvement(tour, distances, forbidden_edges=None):
    """ Find the most fittable edge swap in the current tour"""
    tour_length = len(tour)
    best_improvement = 0
    best_swap = None
    
    if forbidden_edges is None:
        forbidden_edges = set()
    
    for first_idx in range(tour_length):
        city1 = tour[first_idx]
        city2 = tour[(first_idx + 1) % tour_length]
        
        # skip if this edge forbidden
        if create_edge(city1, city2) in forbidden_edges:
            continue
        
        for second_idx in range(tour_length):
            city3 = tour[second_idx]
            city4 = tour[(second_idx + 1) % tour_length]
            
            # Skip if edges are adjacent or forbidden
            are_edges_adjacent = second_idx in [first_idx, 
                                              (first_idx + 1) % tour_length, 
                                              (first_idx - 1) % tour_length]
            if are_edges_adjacent or create_edge(city3, city4) in forbidden_edges:
                continue
                
            # how much we improve by swapping these edges
            improvement = gain_crit(city1, city2, city3, city4, distances)
            
            # best improvement found
            if improvement > best_improvement:
                best_improvement = improvement
                best_swap = (first_idx, second_idx)
    
    return best_swap, best_improvement

def two_opt(tour, move):
    if move is None:
        return tour    
    i,j = move
    if i > j:
        i,j = j,i
    
    new_tour = tour[:i+1] + list(reversed(tour[i+1:j+1])) + tour[j+1:]
    return new_tour

def lin_kernighan(tsp_matrix, max_iterations=1000):
    """
    Lin-Kernighan partial case of 2-opt
    
    Parameters:
    tsp_matrix: distance matrix
    max_iterations: max number of iterations (default 1000)
    
    Returns:
    tuple: best tour found, its cost
    """

    n = len(tsp_matrix)
    current_tour = list(range(n))
    random.shuffle(current_tour)
    
    best_tour = current_tour[:]
    best_cost = route_distance(best_tour, tsp_matrix)
    
    for _ in range(max_iterations):
        improved = False
        tabu_edges = set()
        
        #depth was set to max 5 as if we incrise it 
        #it increases the time complexity even more, so 5 is a good compromise here
        for _ in range(1, min(5, n)): 
            move,gain = find_best_improvement(current_tour, tsp_matrix, tabu_edges)
            
            if move is None or gain <= 0:
                break
                
            current_tour = two_opt(current_tour, move)
            
            i,j = move
            tabu_edges.add(create_edge(current_tour[i], current_tour[(i+1) % n]))
            tabu_edges.add(create_edge(current_tour[j], current_tour[(j+1) % n]))
            
            current_cost = route_distance(current_tour, tsp_matrix)
            
            if current_cost < best_cost:
                best_tour = current_tour[:]
                best_cost = current_cost
                improved = True
        
        #to escape local optima - random restart with 10% probability
        if not improved:
            if random.random() < 0.1:
                current_tour = list(range(n))
                random.shuffle(current_tour)
    
    final_tour = [city + 1 for city in best_tour]
    final_tour.append(final_tour[0])
    
    return final_tour, best_cost

if __name__ == "__main__":

    problem = tsplib95.load('data/ALL_tsp/burma14.tsp')
    distance_matrix = create_distance_matrix(problem)

    start_time = time.time()
    route, total_cost = lin_kernighan(distance_matrix)
    end_time = time.time() - start_time

    print("Best tour:", route)
    print("Cost:", total_cost)
    print(f"Running time: {end_time:.6f} seconds")