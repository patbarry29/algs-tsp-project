import matplotlib.pyplot as plt
import numpy as np
import tsplib95
from data.opt_cost import atsp as opt_sol
from greedy.greedy import greedy
from utils.create_distance_matrix import create_distance_matrix
from utils.get_opt_cost import get_optimal_cost


def plot_relative_error(tsplib_instances, algo_function):
    """
    Plot the relative error between the algorithm's output
    and the optimal costs for multiple TSPLIB instances.

    Parameters:
    tsplib_instances (list): List of TSPLIB instance names.
    algo_function (function): The algorithm function to compute the cost and route.
    """
    relative_errors = []
    instance_names = []

    for instance_name in tsplib_instances:
        try:
            problem = tsplib95.load(f'../data/ALL_atsp/{instance_name}.atsp')
            distance_matrix = create_distance_matrix(problem)
            route, algo_cost = algo_function(distance_matrix)
            optimal_cost = get_optimal_cost(opt_sol.data, instance_name)

            if optimal_cost is not None:
                # Calculate the relative error
                error = (algo_cost - optimal_cost) / optimal_cost * 100
                relative_errors.append(error)
                instance_names.append(instance_name)
            else:
                print(f"Optimal cost not available for instance: {instance_name}")
        except Exception as e:
            print(f"Error processing instance {instance_name}: {e}")

    if relative_errors:
        avg_error = np.mean(relative_errors)

        plt.figure(figsize=(10, 6))
        bars = plt.bar(instance_names, relative_errors)

        # Add the percentage on top of each bar
        for bar, deviation in zip(bars, relative_errors):
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

        plt.axhline(y=avg_error, color='r', linestyle='--', label=f'Average relative error: {avg_error:.2f}%')
        plt.xlabel('TSPLIB Instances')
        plt.ylabel('Relative Error (%)')
        plt.title('Relative Error of Greedy Algorithm')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("No valid relative errors to plot.")

if __name__ == "__main__":
    tsplib_instances = [
        "br17", "ft53", "ft70", "ftv33", "ftv35", "ftv38", "ftv44", "ftv47",
        "ftv55", "ftv64", "ftv70", "ftv170", "p43",
        "rbg323", "rbg358", "rbg403", "rbg443", "ry48p"
    ]
    plot_relative_error(tsplib_instances, greedy)