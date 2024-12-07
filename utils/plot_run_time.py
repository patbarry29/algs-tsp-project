import matplotlib.pyplot as plt
import tsplib95
import time
from greedy.greedy import greedy
from utils.create_distance_matrix import create_distance_matrix


def plot_runtime(tsplib_instances, algo_function):
    """
    Plot the runtime of the algorithm for multiple TSPLIB instances.

    Parameters:
    tsplib_instances (list): List of TSPLIB instance names.
    algo_function (function): The algorithm function to compute the cost and route.
    """
    runtimes = []
    instance_names = []

    for instance_name in tsplib_instances:
        try:
            problem = tsplib95.load(f'../data/ALL_atsp/{instance_name}.atsp')
            distance_matrix = create_distance_matrix(problem)

            start_time = time.time()
            route, algo_cost = algo_function(distance_matrix)
            end_time = time.time()
            runtime = end_time - start_time

            runtimes.append(runtime)
            instance_names.append(instance_name)
            print(f"Processed instance: {instance_name}, Runtime: {runtime:.4f} seconds")

        except Exception as e:
            print(f"Error processing instance {instance_name}: {e}")

    if runtimes:
        plt.figure(figsize=(10, 6))
        plt.plot(instance_names, runtimes, marker='o', linestyle='-')

        for i, runtime in enumerate(runtimes):
            plt.text(i, runtime, f'{runtime:.2f}s', ha='center', va='bottom', fontsize=10, color='black')

        plt.xlabel('TSPLIB Instances')
        plt.ylabel('Runtime (seconds)')
        plt.title('Runtime of Greedy for TSPLIB Instances')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print("No valid runtimes to plot.")


if __name__ == "__main__":
    tsplib_instances = [
        "br17", "ft53", "ft70", "ftv33", "ftv35", "ftv38", "ftv44", "ftv47",
        "ftv55", "ftv64", "ftv70", "ftv170", "p43",
        "rbg323", "rbg358", "rbg403", "rbg443", "ry48p"
    ]
    plot_runtime(tsplib_instances, greedy)
