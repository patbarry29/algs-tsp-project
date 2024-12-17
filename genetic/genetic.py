import random
import time
from random import randint
import matplotlib.pyplot as plt
import numpy as np
import tsplib95
from data.opt_cost import tsp as opt_sol
from utils.create_distance_matrix import create_distance_matrix
from utils.get_opt_cost import get_optimal_cost

# from utils.get_opt_cost import get_optimal_cost
# from data.opt_cost import tsp as opt_sol
# from utils.create_distance_matrix import create_distance_matrix

INT_MAX = 2147483647

def rand_num(start, end):
    """Returns a random number in the range [start, end)."""
    return randint(start, end - 1)

def create_chrom(nb_cities):
    """Creates a valid chromosome (tour) starting and ending at city 0."""
    chrom = [0] + random.sample(range(1, nb_cities), nb_cities - 1)
    chrom.append(0)  # Return to the starting city
    return chrom

def cal_fitness(chrom, distance_matrix):
    """Calculates the fitness value (total cost) of a chromosome."""
    f = 0
    for i in range(len(chrom) - 1):
        city1 = chrom[i]
        city2 = chrom[i + 1]
        if distance_matrix[city1][city2] == INT_MAX:
            return INT_MAX
        f += distance_matrix[city1][city2]
    return f

def initialize_population(nb_cities, pop_size, distance_matrix):
    """Creates the initial population of individuals."""
    population = []
    for _ in range(pop_size):
        chrom = create_chrom(nb_cities)
        fitness = cal_fitness(chrom, distance_matrix)
        population.append({'chrom': chrom, 'fitness': fitness})
    return population

def selection_tournament(population, tourn_size=3):
    """Selects the best individuals from a population using tournament selection."""
    pop_size = len(population)
    fitness_array = np.array([ind['fitness'] for ind in population])
    aspirants_idx = np.random.randint(pop_size, size=(pop_size, tourn_size))  # Select indices for tournaments
    aspirants_values = fitness_array[aspirants_idx]  # Get fitness values of aspirants
    winner_idx = aspirants_values.argmin(axis=1)  # Index of the winner for each tournament (lower fitness is better)
    sel_index = [aspirants_idx[i, j] for i, j in enumerate(winner_idx)]  # Index of the selected individuals
    selected_population = [population[idx] for idx in sel_index]  # Get the winners from the original population
    return selected_population

def apply_crossover(parent1, parent2, distance_matrix):
    """Applies Order Crossover (OX1) ensuring the starting city remains the same."""
    size = len(parent1['chrom']) - 1
    child_chrom = [-1] * size
    child_chrom[0] = parent1['chrom'][0]  # Keep the starting city
    start, end = sorted([rand_num(1, size), rand_num(1, size)])
    child_chrom[start:end] = parent1['chrom'][start:end]
    p2_genes = [gene for gene in parent2['chrom'] if gene not in child_chrom and gene != child_chrom[0]]
    i = 0
    for j in range(1, size):  # Start from 1 to avoid changing the starting city
        if child_chrom[j] == -1:
            child_chrom[j] = p2_genes[i]
            i += 1
    child_chrom.append(child_chrom[0])
    child_fitness = cal_fitness(child_chrom, distance_matrix)
    return {'chrom': child_chrom, 'fitness': child_fitness}
def apply_mutation(population, mutation_rate, distance_matrix):
    """Applies reverse mutation without altering the starting city."""
    for individual in population:
        if random.random() < mutation_rate:
            Chrom = np.array(individual['chrom'])
            n1, n2 = np.random.randint(1, len(Chrom) - 1, 2)  # Start from index 1
            if n1 >= n2:
                n1, n2 = n2, n1 + 1
            Chrom[n1:n2] = Chrom[n1:n2][::-1]
            individual['chrom'] = list(Chrom)
            individual['fitness'] = cal_fitness(individual['chrom'], distance_matrix)
    return population

def plot_best_individual(best_fitness):
    generations = list(range(1, len(best_fitness) + 1))
    plt.plot(generations, best_fitness, color='g', label='Best Fitness')
    plt.title('Best Fitness over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid()
    plt.show()

def genetic(distance_matrix, hyperparams=None):

    defaults = {
        "POP_SIZE": 200,
        "GEN_THRESH": 500,
        "crossover_rate": 0.9,
        "mutation_rate": 0.1,
    }
    if hyperparams:
        defaults.update(hyperparams)

    nb_cities = len(distance_matrix)
    pop_size = defaults["POP_SIZE"]
    gen_thresh = defaults["GEN_THRESH"]
    crossover_rate = defaults["crossover_rate"]
    mutation_rate = defaults["mutation_rate"]

    best_fitness = []

    # Initialize population
    population = initialize_population(nb_cities, pop_size, distance_matrix)

    # Iterate through generations
    for gen in range(gen_thresh):
        population.sort(key=lambda x: x['fitness'])

        current_best_fitness = population[0]['fitness']
        best_fitness.append(current_best_fitness)

        selected_population = selection_tournament(population)
        new_population = []
        for i in range(0, len(selected_population), 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i + 1] if i + 1 < len(selected_population) else parent1
            if random.random() < crossover_rate:
                child = apply_crossover(parent1, parent2, distance_matrix)
                new_population.append(child)

        new_population.extend(selected_population)  # Combine parents and children
        new_population = sorted(new_population, key=lambda x: x['fitness'])[:pop_size]  # Select the best individuals

        new_population = apply_mutation(new_population, mutation_rate, distance_matrix)

        population = new_population

    best_individual = min(population, key=lambda x: x['fitness'])
    cost = best_individual['fitness']
    seq = [int(x + 1) for x in best_individual['chrom']]
    return seq, cost

def plot_fitness_vs_generation(distance_matrix):
    """Plots generation threshold vs fitness for different mutation rates."""
    crossover_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    gen_thresh = range(0, 101, 10)

    for rate in crossover_rates:
        fitness = []
        for thresh in gen_thresh:
            _, cost = genetic(distance_matrix, {"GEN_THRESH": thresh, "mutation_rate": 0.1, "crossover_rate": 0.9})
            fitness.append(cost)
        plt.plot(gen_thresh, fitness, label=f'Crossover Rate: {rate}', marker='x', alpha=0.7)
    plt.title('TSPLIB instance eil51')
    plt.xlabel('Generation Threshold')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid()
    plt.show()

# def plot_computation_time_vs_problem():
#     """Plots computational time for different TSPLIB instances over various generation thresholds."""
#     data = ['burma14', 'bayg29', 'eil51']
#     gen_thresh = [100, 500, 900]
#     computation_times = {thresh: [] for thresh in gen_thresh}
#
#     for d in data:
#         for thresh in gen_thresh:
#             start_time = time.time()
#             problem = tsplib95.load(f'../data/ALL_tsp/{d}.tsp')
#             distance_matrix = create_distance_matrix(problem)
#             _, _ = genetic(distance_matrix, {"GEN_THRESH": thresh})
#             end_time = time.time()
#             computation_times[thresh].append((end_time - start_time) / 1000)
#
#     for thresh in gen_thresh:
#         plt.plot(data, computation_times[thresh], label=f'Generation Threshold: {thresh}', marker='o')
#
#     plt.title('Computation Time vs Problem')
#     plt.xlabel('Problem')
#     plt.ylabel('Time (s)')
#     plt.legend()
#     plt.grid()
#     plt.show()

def plot_computation_time_vs_problem():
    """Plots computational time for different TSPLIB instances over various generation thresholds."""
    data = ['burma14', 'bayg29', 'eil51']
    gen_thresh = [100, 500, 1000]
    computation_times = {d: [] for d in data}

    for d in data:
        for thresh in gen_thresh:
            start_time = time.time()
            problem = tsplib95.load(f'../data/ALL_tsp/{d}.tsp')
            distance_matrix = create_distance_matrix(problem)
            _, _ = genetic(distance_matrix, {"GEN_THRESH": thresh})
            end_time = time.time()
            computation_times[d].append(end_time - start_time)

    for d in data:
        plt.plot(gen_thresh, computation_times[d], label=f'{d}')

    plt.title('Computation Time vs Generation Threshold')
    plt.xlabel('Generation Threshold')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.grid()
    plt.show()


def gen_thresh_pop_size_vs_computation_time():
    test_range = [100, 500, 1000, 5000]
    computation_times_gen_thresh = []
    computation_times_pop_size = []

    for value in test_range:
        # Measure computation time for gen_thresh
        start_time = time.time()
        problem = tsplib95.load(f'../data/ALL_tsp/burma14.tsp')
        distance_matrix = create_distance_matrix(problem)
        route, algo_cost = genetic(distance_matrix, {"GEN_THRESH": value})
        computation_time = time.time() - start_time
        computation_times_gen_thresh.append(computation_time)

        # Measure computation time for pop_size
        start_time = time.time()
        route, algo_cost = genetic(distance_matrix, {"POP_SIZE": value})
        computation_time = time.time() - start_time
        computation_times_pop_size.append(computation_time)

    plt.figure(figsize=(6, 6))
    plt.plot(test_range, computation_times_gen_thresh, marker='o', label='GEN_THRESH', linestyle='-')
    plt.plot(test_range, computation_times_pop_size, marker='o', label='POP_SIZE', linestyle='-')

    plt.xlabel('Test Range Values')
    plt.ylabel('Computation Time (s)')
    plt.title('Computation Time vs GEN_THRESH and POP_SIZE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


if __name__ == '__main__':
    gen_thresh_pop_size_vs_computation_time()
    # data = "eil51"
    # problem = tsplib95.load(f'../data/ALL_tsp/{data}.tsp')
    # # problem = tsplib95.load('../data/random/tsp/random_tsp5.tsp')
    # distance_matrix = create_distance_matrix(problem)

    # hyperparams = {
    #     "POP_SIZE": 200,
    #     "GEN_THRESH": 5000,
    #     "crossover_rate": 0.9,
    #     "mutation_rate": 0.3,
    # }

    # plot_fitness_vs_generation(distance_matrix)
    # plot_computation_time_vs_problem()

    # cost, seq = genetic(distance_matrix, {"GEN_THRESH": 5000})
    # print('\nCost:', cost)
    # print('Sequence:', seq)
    #
    # opt_cost = get_optimal_cost(opt_sol.data, data)
    # print("\nOptimal cost:", opt_cost)
    # print("\nRelative error:", (cost - opt_cost) / opt_cost)


