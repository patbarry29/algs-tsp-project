import os
import sys
import time
import tsplib95

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.create_distance_matrix import create_distance_matrix

def lin_kernighan(tsp_matrix):
    n = len(tsp_matrix)
    current_tour = list(range(n))
    best_tour = current_tour[:]
    best_cost = route_distance(best_tour, tsp_matrix)

    improvement_found = True
    while improvement_found:
        improvement_found = False
        
        for i in range(n - 1):
            for j in range(i + 2, n):
                old_distance = (tsp_matrix[current_tour[i]][current_tour[i+1]] + 
                              tsp_matrix[current_tour[j]][current_tour[(j+1) % n]])
                new_distance = (tsp_matrix[current_tour[i]][current_tour[j]] + 
                              tsp_matrix[current_tour[i+1]][current_tour[(j+1) % n]])
                
                if new_distance < old_distance:
                    current_tour[i+1:j+1] = reversed(current_tour[i+1:j+1])
                    new_cost = route_distance(current_tour, tsp_matrix)
                    
                    if new_cost < best_cost:
                        best_tour = current_tour[:]
                        best_cost = new_cost
                        improvement_found = True
                        break
            
            if improvement_found:
                break

    final_tour = [city + 1 for city in best_tour]
    final_tour.append(final_tour[0])

    return final_tour, best_cost

def route_distance(tour, tsp_matrix):
    cost = 0
    for i in range(len(tour)-1):
        cost += tsp_matrix[tour[i]][tour[i+1]]
    cost += tsp_matrix[tour[-1]][tour[0]]
    return cost

if __name__ == "__main__":
    problem = tsplib95.load('data/ALL_tsp/burma14.tsp')
    distance_matrix = create_distance_matrix(problem)

    start_time = time.time()
    route, total_cost = lin_kernighan(distance_matrix)
    end_time = time.time() - start_time

    print("Best tour:", route)
    print("Cost:", total_cost)
    print(f"Running time: {end_time:.6f} seconds")