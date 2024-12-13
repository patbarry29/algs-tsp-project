import matplotlib.pyplot as plt
import numpy as np
import tsplib95
from data.opt_cost import tsp as opt_sol
from greedy.greedy import greedy
from utils.create_distance_matrix import create_distance_matrix
from utils.get_opt_cost import get_optimal_cost


def plot_deviation(tsplib_instances, algo_function):
    """
    Plot the deviation between the algorithm's output
    and the optimal costs for multiple TSPLIB instances.

    Parameters:
    tsplib_instances (list): List of TSPLIB instance names.
    algo_function (function): The algorithm function to compute the cost and route.
    """
    deviations = []
    instance_names = []

    for instance_name in tsplib_instances:
        try:
            problem = tsplib95.load(f'../data/ALL_tsp/{instance_name}.tsp')
            distance_matrix = create_distance_matrix(problem)
            route, algo_cost = algo_function(distance_matrix)
            optimal_cost = get_optimal_cost(opt_sol.data, instance_name)

            if optimal_cost is not None:
                # Calculate the deviation in percentage
                error = (algo_cost - optimal_cost) / optimal_cost * 100
                deviations.append(error)
                instance_names.append(instance_name)
            else:
                print(f"Optimal cost not available for instance: {instance_name}")
        except Exception as e:
            print(f"Error processing instance {instance_name}: {e}")

    if deviations:
        avg_error = np.mean(deviations)

        plt.figure(figsize=(6, 6))
        bars = plt.bar(instance_names, deviations)

        # Add the percentage on top of each bar
        for bar, deviation in zip(bars, deviations):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f'{deviation:.2f}%',
                ha='center',
                va='bottom',
                fontsize=10,
                color='black'
            )

        plt.axhline(y=avg_error, color='r', linestyle='--', label=f'Average deviation: {avg_error:.2f}%')
        plt.xlabel('TSPLIB Instances')
        plt.ylabel('Deviations (%)')
        plt.title('Deviations of Greedy Algorithm')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("No valid deviation to plot.")

if __name__ == "__main__":
    tsplib_instances = [
        "berlin52",
        "ch150",
        "pr1002",
        "pla7397"
    ]
    plot_deviation(tsplib_instances, greedy)