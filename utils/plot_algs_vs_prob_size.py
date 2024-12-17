# IMPORTS
import tsplib95
import matplotlib.pyplot as plt
import numpy as np
import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)


from dynamic_programming.dynamic_programming import dynamic_programming
from genetic.genetic import genetic
from branch_and_bound.branch_and_bound import branch_and_bound as bb1
from branch_and_bound.reduction_matrix_edge_selection import branch_and_bound1 as bb2
from greedy.greedy import greedy
from lin_kernighan.lin_kernighan import lin_kernighan
from randomized.randomized import randomized
from ant_colony.ant_colony import ant_colony
from brute_force.brute_force import brute_force
from mst.mst import mst

from utils.create_distance_matrix import create_distance_matrix
from utils.generate_tsp import generate_tsp
from performance import compute_cpu_usage


def plot_algs_vs_problem_size(tsp_solvers, solver_names, problem_sizes, measure='deviation'):
    if len(tsp_solvers) != len(solver_names):
        raise ValueError("Number of solvers must match number of solver names")

    np.random.seed(np.random.randint(10))
    avg_metrics_per_solver = [[] for _ in range(len(tsp_solvers))]
    problem_sizes_per_solver = [[] for _ in range(len(tsp_solvers))]

    for num_cities in problem_sizes:
        print('num cities:',num_cities)
        for _ in range(5):  # Create 30 random problems for each number of cities
            cities = generate_tsp(num_cities, dim_size=100)
            problem = tsplib95.load(os.path.join("data","random","tsp","random_tsp.tsp"))
            distance_matrix = create_distance_matrix(problem)

            problem_metrics = []
            for i, solver in enumerate(tsp_solvers):
                if solver.__name__ == 'brute_force' and num_cities > 11:
                    problem_metrics.append(None)
                    continue
                if solver == bb1 and num_cities > 11:
                    problem_metrics.append(None)
                    continue
                if solver.__name__ == 'dynamic_programming' and num_cities > 16:
                    problem_metrics.append(None)
                    continue
                if measure == 'deviation':
                    _, total_cost = solver(distance_matrix)
                    total_cost = np.round(result[1], 2)
                    problem_metrics.append(total_cost)
                elif measure == 'time':
                    _, running_time, _ = compute_cpu_usage(solver, distance_matrix)
                    problem_metrics.append(running_time)
                elif measure == 'both':
                    result, running_time, _ = compute_cpu_usage(solver, distance_matrix)
                    total_cost = np.round(result[1], 2)
                    problem_metrics.append((total_cost, running_time))

            if measure == 'deviation':
                min_cost_for_problem = np.min(np.asarray([x for x in problem_metrics if x is not None]))
            elif measure == 'both':
                min_cost_for_problem = np.min(np.asarray([x for x in problem_metrics if x is not None])[:, 0])

            for i, solver_metrics in enumerate(problem_metrics):
                if solver_metrics is None:
                    continue

                if measure == 'deviation':
                    print(solver_metrics)
                    solver_metrics = ((solver_metrics - min_cost_for_problem) / min_cost_for_problem)*100
                    print(solver_metrics)
                elif measure == 'both':
                    solver_metrics = (
                        (solver_metrics[0] - min_cost_for_problem) / min_cost_for_problem, solver_metrics[1])*100

                avg_metrics_per_solver[i].append(solver_metrics)
                problem_sizes_per_solver[i].append(num_cities)

    averaged_metrics_per_solver = []
    unique_problem_sizes = sorted(set(problem_sizes))

    for i, solver_metrics in enumerate(avg_metrics_per_solver):
        solver_problem_sizes = problem_sizes_per_solver[i]
        solver_avg_metrics = []
        for size in unique_problem_sizes:
            indices = [idx for idx, s in enumerate(solver_problem_sizes) if s == size]
            if indices:
                metrics = [solver_metrics[idx] for idx in indices]
                if measure == 'both':
                    avg_deviation = np.mean([m[0] for m in metrics])*100
                    avg_time = np.mean([m[1] for m in metrics])
                    solver_avg_metrics.append((size, (avg_deviation, avg_time)))
                else:
                    avg_metric = np.mean(metrics)
                    if measure=='deviation':
                        avg_metric *= 100
                    solver_avg_metrics.append((size, avg_metric))
        averaged_metrics_per_solver.append(solver_avg_metrics)

    plt.figure(figsize=(6, 6))

    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 1, 1))]
    colors = plt.cm.tab10.colors
    alphas = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    if measure == 'both':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        for i, solver_metrics in enumerate(averaged_metrics_per_solver):
            problem_sizes = [size for size, _ in solver_metrics]
            avg_deviations = [metrics[0] for _, metrics in solver_metrics]
            avg_times = [metrics[1] for _, metrics in solver_metrics]

            line_style = line_styles[i % len(line_styles)]
            color = colors[i % len(colors)]
            alpha = alphas[i % len(alphas)]

            ax1.plot(problem_sizes, avg_deviations,
                    color=color, linestyle=line_style, alpha=alpha, label=f"{solver_names[i]} Deviation", marker='o')
            ax2.plot(problem_sizes, avg_times,
                    color=color, linestyle=line_style, alpha=alpha, label=f"{solver_names[i]} Time", marker='o')
            for j, txt in enumerate(avg_times):
                ax2.annotate(f"{txt:.2f}", (problem_sizes[j], avg_times[j]), textcoords="offset points", xytext=(0, 5),
                            ha='center')

        ax1.set_xlabel("Number of Cities")
        ax1.set_ylabel("Average Total Deviation (%)")
        ax1.set_title("Average Deviation vs. Number of Cities in TSP")
        ax1.legend()

        ax2.set_xlabel("Number of Cities")
        ax2.set_ylabel("Average Execution Time (s)")
        ax2.set_title("Average Execution Time vs. Number of Cities in TSP")
        ax2.legend()
    else:
        for i, solver_metrics in enumerate(averaged_metrics_per_solver):
            problem_sizes = [size for size, _ in solver_metrics]
            avg_metrics = [metrics for _, metrics in solver_metrics]

            line_style = line_styles[i % len(line_styles)]
            color = colors[i % len(colors)]
            alpha = alphas[i % len(alphas)]

            plt.plot(problem_sizes, avg_metrics,
                    color=color, linestyle=line_style, marker='o', alpha=alpha, label=solver_names[i])
            if measure == 'time':
                for j, txt in enumerate(avg_metrics):
                    plt.annotate(f"{txt:.2f}", (problem_sizes[j], avg_metrics[j]), textcoords="offset points",
                                xytext=(0, 5),
                                ha='center')

        plt.xlabel("Number of Cities")
        plt.ylabel("Average Total Deviation (%)" if measure == 'deviation' else "Average Execution Time (s)")
        plt.title("Average Deviation vs. Number of Cities in TSP" if measure == 'deviation' else "Average Execution Time vs. Number of Cities in TSP")
        plt.legend()

    plt.tight_layout()
    plt.show()

    plt.style.use('seaborn-v0_8')
    plt.rcParams['axes.facecolor'] = 'white'


if __name__ == '__main__':
    algorithms = [brute_force]
    algorithms_names = ['Brute Force']
    problem_sizes = list(range(4, 12, 1))
    plot_algs_vs_problem_size(algorithms, algorithms_names, problem_sizes, measure='time')

