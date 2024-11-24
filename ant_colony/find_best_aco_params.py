import numpy as np
import tsplib95
import time
import pandas as pd
import matplotlib.pyplot as plt
from ant_colony import ant_colony
from generate_tsp import generate_tsp
from create_distance_matrix import create_distance_matrix

def test_parameter_values(param_name, param_values, problem_sizes, num_problems_per_size):
    """Test different values for a specified parameter."""
    results = []

    base_params = {
        'alpha': 1,
        'beta': 2,
        'num_iterations': 100,
        'evaporation_rate': 0.1,
        'initial_pheromone_level': 0.1,
        'num_ants': 50,
        'mute_output': True
    }

    for param_value in param_values:
        print(f"\nTesting {param_name} = {param_value}")

        # Update parameter value
        test_params = base_params.copy()
        test_params[param_name] = param_value

        for size in problem_sizes:
            distances = []
            times = []

            for problem_num in range(num_problems_per_size):
                # Generate and solve problem
                generate_tsp(n=size)
                problem = tsplib95.load('random_tsp.tsp')
                distance_matrix = create_distance_matrix(problem)

                start_time = time.time()
                _, best_distance = ant_colony(distance_matrix, test_params)
                solve_time = time.time() - start_time

                distances.append(best_distance)
                times.append(solve_time)

            # Record results for this problem size
            results.append({
                'param_value': param_value,
                'problem_size': size,
                'avg_distance': np.mean(distances),
                'std_distance': np.std(distances),
                'avg_time': np.mean(times)
            })

    return pd.DataFrame(results)

def plot_results(df):
    """Plot the results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot average distances
    for size in df['problem_size'].unique():
        size_data = df[df['problem_size'] == size]
        ax1.plot(size_data['param_value'], size_data['avg_distance'],
                label=f'Size {size}', marker='o')

    ax1.set_xlabel('Parameter Value')
    ax1.set_ylabel('Average Distance')
    ax1.set_title('Parameter Value vs Average Distance')
    ax1.legend()

    # Plot average times
    for size in df['problem_size'].unique():
        size_data = df[df['problem_size'] == size]
        ax2.plot(size_data['param_value'], size_data['avg_time'],
                label=f'Size {size}', marker='o')

    ax2.set_xlabel('Parameter Value')
    ax2.set_ylabel('Average Time (s)')
    ax2.set_title('Parameter Value vs Average Time')
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test configuration
    param_name = 'num_ants'
    param_values = np.arange(10,101,10)
    problem_sizes = np.arange(10,101,10)
    num_problems_per_size = 10

    # Run tests
    results_df = test_parameter_values(param_name, param_values, problem_sizes, num_problems_per_size)

    # Print results
    print("\nDetailed Results:")
    print(results_df)

    # Find best parameter value
    avg_performance = results_df.groupby('param_value')['avg_distance'].mean()
    best_value = avg_performance.idxmin()
    print(f"\nBest {param_name} value: {best_value}")
    print(f"Average distances for each value:")
    print(avg_performance)

    # Plot results
    plot_results(results_df)