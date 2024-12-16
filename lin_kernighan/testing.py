import os
import sys
import time
import tsplib95
import matplotlib.pyplot as plt
from tabulate import tabulate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.create_distance_matrix import create_distance_matrix
from lin_kernighan import lin_kernighan as lk_solver
from two_opt import lin_kernighan as opt2_solver

# Test instances with their optimal solutions
TEST_INSTANCES = {
    'burma14.tsp': 3323,    # 14 cities
    'fri26.tsp': 937,       # 26 cities
    'att48.tsp': 10628,     # 48 cities
    'berlin52.tsp': 7542,   # 52 cities
    'st70.tsp': 675,        # 70 cities
    'kroA100.tsp': 21282    # 100 cities
}

def run_benchmark(instance_name, solver):
    problem = tsplib95.load(f'data/ALL_tsp/{instance_name}')
    distance_matrix = create_distance_matrix(problem)
    
    start_time = time.time()
    route, cost = solver(distance_matrix)
    end_time = time.time() - start_time
    
    return cost, end_time

def plot_results(results):
    # Extract data for plotting
    instances = [row[0].split('.')[0] for row in results]
    cities = [int(''.join(filter(str.isdigit, inst))) for inst in instances]
    opt2_times = [float(row[2]) for row in results]
    lk_times = [float(row[4]) for row in results]
    opt2_costs = [float(row[1]) for row in results]
    lk_costs = [float(row[3]) for row in results]
    optimal_costs = [TEST_INSTANCES[inst] for inst in TEST_INSTANCES]

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Plot execution times
    ax1.plot(cities, opt2_times, 'bo-', label='2-opt')
    ax1.plot(cities, lk_times, 'ro-', label='Lin-Kernighan')
    ax1.set_xlabel('Number of Cities')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Execution Time vs Problem Size')
    ax1.legend()
    ax1.grid(True)

    # Plot solution quality (costs)
    ax2.plot(cities, opt2_costs, 'bo-', label='2-opt')
    ax2.plot(cities, lk_costs, 'ro-', label='Lin-Kernighan')
    ax2.plot(cities, optimal_costs, 'go-', label='Optimal')
    ax2.set_xlabel('Number of Cities')
    ax2.set_ylabel('Solution Cost')
    ax2.set_title('Solution Quality vs Problem Size')
    ax2.legend()
    ax2.grid(True)

    # Plot relative error
    opt2_errors = [(c - o) / o * 100 for c, o in zip(opt2_costs, optimal_costs)]
    lk_errors = [(c - o) / o * 100 for c, o in zip(lk_costs, optimal_costs)]
    ax3.plot(cities, opt2_errors, 'bo-', label='2-opt')
    ax3.plot(cities, lk_errors, 'ro-', label='Lin-Kernighan')
    ax3.set_xlabel('Number of Cities')
    ax3.set_ylabel('Error from Optimal (%)')
    ax3.set_title('Relative Error vs Problem Size')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig('comparison_results.png')
    plt.show()

def main():
    results = []
    
    for instance, optimal in TEST_INSTANCES.items():
        print(f"Processing {instance}...")
        
        # Run 2-opt
        opt2_cost, opt2_time = run_benchmark(instance, opt2_solver)
        
        # Run Lin-Kernighan
        lk_cost, lk_time = run_benchmark(instance, lk_solver)
        
        results.append([
            instance,
            opt2_cost,
            f"{opt2_time:.3f}",
            lk_cost,
            f"{lk_time:.3f}",
            f"{((opt2_cost - lk_cost) / lk_cost * 100):.2f}%",
            optimal,
            f"{((opt2_cost - optimal) / optimal * 100):.2f}%",
            f"{((lk_cost - optimal) / optimal * 100):.2f}%"
        ])
    
    headers = [
        "Instance", 
        "2-opt Cost", 
        "2-opt Time(s)", 
        "LK Cost", 
        "LK Time(s)",
        "Cost Diff %",
        "Optimal",
        "2-opt vs Opt %",
        "LK vs Opt %"
    ]
    
    print("\nResults:")
    print(tabulate(results, headers=headers, tablefmt="grid"))

    # Plot the results
    plot_results(results)

if __name__ == "__main__":
    main()