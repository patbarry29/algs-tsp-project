import matplotlib.pyplot as plt
import numpy as np
import tsplib95
from data.opt_cost import tsp as opt_sol
from genetic.genetic import genetic
from greedy.greedy import greedy
from utils.create_distance_matrix import create_distance_matrix
from utils.get_opt_cost import get_optimal_cost


def plot_deviation(tsplib_instances, algo_function):
    """
    Plot the deviation between the algorithm's output
    and the optimal costs for multiple TSPLIB instances.

    Params:
    tsplib_instances (list)
    algo_function (function)
    """
    deviations = []
    instance_names = []

    for instance_name in tsplib_instances:
        problem = tsplib95.load(f'../data/ALL_tsp/{instance_name}.tsp')
        print(f"Solving for instance: {instance_name}")
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


def plot_deviation_genetic_vs_pop_size(tsplib_instances):
    deviations = {100: [], 500: [], 1000: [], 5000: []}
    instance_names = tsplib_instances

    for instance_name in tsplib_instances:
        problem = tsplib95.load(f'../data/ALL_tsp/{instance_name}.tsp')
        print(f"Solving for instance: {instance_name}")
        distance_matrix = create_distance_matrix(problem)

        pop_sizes = [100, 500, 1000, 5000]
        for pop_size in pop_sizes:
            print(f"Running genetic algorithm with population size: {pop_size}")
            route, algo_cost = genetic(distance_matrix, {"POP_SIZE": pop_size})
            optimal_cost = get_optimal_cost(opt_sol.data, instance_name)

            if optimal_cost is not None:
                error = (algo_cost - optimal_cost) / optimal_cost * 100
                deviations[pop_size].append(error)
            else:
                print(f"Optimal cost not available for instance: {instance_name}")

    if any(deviations.values()):
        avg_error = np.mean([error for pop_size in deviations for error in deviations[pop_size]])

        x = np.arange(len(instance_names))  # the label locations
        width = 0.2  # the width of the bars

        plt.figure(figsize=(6, 6))

        for i, pop_size in enumerate([100, 500, 1000, 5000]):
            deviations_for_pop_size = deviations[pop_size]
            x_positions = x + (i - 1) * width  # shift for each population size
            bars = plt.bar(x_positions, deviations_for_pop_size, width=width, label=f'Pop {pop_size}')

            for bar, deviation in zip(bars, deviations_for_pop_size):
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

        # plt.axhline(y=avg_error, color='r', linestyle='--', label=f'Average deviation: {avg_error:.2f}%')
        plt.xlabel('TSPLIB Instances')
        plt.ylabel('Deviations (%)')
        plt.title('Deviations of Genetic Algorithm for Different Population Sizes')
        plt.xticks(x, instance_names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("No valid deviation to plot.")

def plot_deviation_genetic_vs_gen_thresh(tsplib_instances):

    deviations = {100: [], 500: [], 1000: [], 5000: []}
    instance_names = tsplib_instances

    for instance_name in tsplib_instances:
        problem = tsplib95.load(f'../data/ALL_tsp/{instance_name}.tsp')
        print(f"Solving for instance: {instance_name}")
        distance_matrix = create_distance_matrix(problem)

        gen_thresh = [100, 500, 1000, 5000]
        for thresh in gen_thresh:
            print(f"Running genetic algorithm with generation threshold: {thresh}")
            route, algo_cost = genetic(distance_matrix, {"GEN_THRESH": thresh})
            optimal_cost = get_optimal_cost(opt_sol.data, instance_name)

            if optimal_cost is not None:
                # Calculate the deviation in percentage
                error = (algo_cost - optimal_cost) / optimal_cost * 100
                deviations[thresh].append(error)
            else:
                print(f"Optimal cost not available for instance: {instance_name}")

    if any(deviations.values()):
        avg_error = np.mean([error for thresh in deviations for error in deviations[thresh]])

        x = np.arange(len(instance_names))
        width = 0.2
        plt.figure(figsize=(6, 6))

        for i, gen_thresh in enumerate([100, 500, 1000, 5000]):
            deviations_for_gen_thresh = deviations[gen_thresh]
            x_positions = x + (i - 1) * width  # shift for each population size
            bars = plt.bar(x_positions, deviations_for_gen_thresh, width=width, label=f'Gen thresh {gen_thresh}')

            for bar, deviation in zip(bars, deviations_for_gen_thresh):
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

        # plt.axhline(y=avg_error, color='r', linestyle='--', label=f'Average deviation: {avg_error:.2f}%')
        plt.xlabel('TSPLIB Instances')
        plt.ylabel('Deviations (%)')
        plt.title('Deviations of Genetic Algorithm for Different Generation Thresholds')
        plt.xticks(x, instance_names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("No valid deviation to plot.")


if __name__ == "__main__":
    tsplib_instances = [
        "burma14",
        "bayg29",
        "eil51",
    ]

    # plot_deviation_genetic_vs_pop_size(tsplib_instances)
    plot_deviation_genetic_vs_gen_thresh(tsplib_instances)