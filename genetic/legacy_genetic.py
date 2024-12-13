import random
import time
from random import randint, shuffle

import numpy as np
import tsplib95
from utils.get_opt_cost import get_optimal_cost
from data.opt_cost import tsp as opt_sol
from utils.create_distance_matrix import create_distance_matrix
import optuna
import plotly
import sklearn

INT_MAX = 2147483647

class Individual:
    """Class to represent an individual in the population (GNOME or tour)."""
    def __init__(self) -> None:
        self.gnome = []
        self.fitness = 0

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __gt__(self, other):
        return self.fitness > other.fitness


def rand_num(start, end):
    """Returns a random number in the range [start, end)."""
    return randint(start, end - 1)


def create_gnome(nb_cities):
    """Creates a valid GNOME (tour)."""
    gnome = list(range(nb_cities))
    gnome = random.sample(gnome, len(gnome))  # Shuffle the list
    gnome.append(gnome[0])  # Return to the starting city
    return gnome


def mutated_gene(gnome, nb_cities):
    """Randomly swaps two cities in the GNOME."""
    gnome = gnome.copy()

    # e.g, if nb_cities = 5, consider indices [2, 3, 4] (excluding first and last city)
    r, r1 = random.sample(range(1, nb_cities - 1), 2)

    # Swap the two selected cities
    gnome[r], gnome[r1] = gnome[r1], gnome[r]

    # Ensure the last city is equal to the first city to maintain cyclic structure
    if gnome[-1] != gnome[0]:
        gnome[-1] = gnome[0]

    return gnome


def cal_fitness(gnome, distance_matrix):
    """Calculates the fitness value (total cost) of a GNOME."""
    f = 0
    for i in range(len(gnome) - 1):
        city1 = gnome[i]
        city2 = gnome[i + 1]
        if distance_matrix[city1][city2] == INT_MAX:
            return INT_MAX
        f += distance_matrix[city1][city2]
    return f


# def cooldown(temp):
#     """Reduces the temperature."""
#     return temp * 0.95  # Reduce temperature by 5% per generation


# def two_opt(gnome, distance_matrix):
#     """
#     Applies the 2-opt local search to refine a GNOME (tour).
#     """""
#     # Initialize the best tour as the current GNOME
#     best = gnome[:]
#     # Calculate the cost of the current best tour
#     best_cost = cal_fitness(best, distance_matrix)
#     # Set a flag to track if improvements are made
#     improved = True
#
#     # Continue searching for improvements until no changes are found
#     while improved:
#         improved = False  # Reset the improvement flag for this iteration
#
#         # Loop over all possible pairs of edges in the tour
#         for i in range(1, len(best) - 2):  # Start from the second city (index 1)
#             for j in range(i + 1, len(best) - 1):  # Ensure j > i to avoid duplicate checks
#                 if j - i == 1:  # Skip adjacent nodes, as swapping adjacent edges is unnecessary
#                     continue
#
#                 # Create a new GNOME by reversing the segment between i and j
#                 new_gnome = best[:]  # Copy the current best tour
#                 new_gnome[i:j] = best[j - 1:i - 1:-1]  # Reverse the segment between indices i and j
#
#                 # Calculate the cost of the new GNOME
#                 new_cost = cal_fitness(new_gnome, distance_matrix)
#
#                 # If the new GNOME has a lower cost, update the best tour
#                 if new_cost < best_cost:
#                     best = new_gnome[:]  # Update the best tour
#                     best_cost = new_cost  # Update the best cost
#                     improved = True  # Set the improvement flag to True
#
#     # Return the refined tour after no further improvements are found
#     return best


def initialize_population(nb_cities, pop_size, distance_matrix):
    """Creates the initial population of individuals."""
    population = []
    for _ in range(pop_size):
        gnome = create_gnome(nb_cities)
        fitness = cal_fitness(gnome, distance_matrix)
        individual = Individual()
        individual.gnome = gnome
        individual.fitness = fitness
        population.append(individual)
    return population


def select_parent(population):
    """Selects a parent using tournament selection."""
    tournament_size = 5  # Adjust as needed
    tournament = random.sample(population, tournament_size)
    parent = min(tournament, key=lambda x: x.fitness)
    return parent


def apply_crossover(parent1, parent2, distance_matrix):
    """Applies Order Crossover (OX1) to generate offspring."""
    size = len(parent1.gnome) - 1  # Do not include the last repeated city
    child_gnome = [-1] * size  # Initialize child with placeholders

    # Select two random crossover points
    start, end = sorted([rand_num(1, size), rand_num(1, size)])

    # Copy sub-tour from parent1 to child
    child_gnome[start:end] = parent1.gnome[start:end]

    # Fill remaining positions using parent2 while maintaining order
    p2_genes = [gene for gene in parent2.gnome if gene not in child_gnome]

    # Step 4: Fill in the missing values in child_gnome
    i = 0
    for j in range(size):
        if child_gnome[j] == -1:  # If the position is not filled
            child_gnome[j] = p2_genes[i]  # Fill with the next available gene from parent2
            i += 1

    # Add the starting city at the end to complete the cycle
    child_gnome.append(child_gnome[0])

    child_fitness = cal_fitness(child_gnome, distance_matrix)

    child = Individual()
    child.gnome = child_gnome
    child.fitness = child_fitness

    return child


# def apply_mutation(individual, mutation_rate, distance_matrix):
#     """Applies Swap Mutation to an individual."""
#     if random.random() < mutation_rate:
#         mutated_gnome = mutated_gene(individual.gnome, len(distance_matrix))
#         individual.gnome = mutated_gnome
#         individual.fitness = cal_fitness(mutated_gnome, distance_matrix)
#     return individual


# def refine_population(population, distance_matrix):
#     """Applies 2-opt refinement to improve solutions in the population."""
#     for individual in population:
#         refined_gnome = two_opt(individual.gnome, distance_matrix)
#         individual.gnome = refined_gnome
#         individual.fitness = cal_fitness(refined_gnome, distance_matrix)
#     return population


def genetic(distance_matrix, hyperparams=None):

    # Default hyperparameters
    defaults = {
        "POP_SIZE": 200,
        "GEN_THRESH": 500,
        "cooling_rate": 0.95,
        "crossover_rate": 0.8,
        "mutation_rate": 0.2,
    }
    if hyperparams:
        defaults.update(hyperparams)

    nb_cities = len(distance_matrix)
    pop_size = defaults["POP_SIZE"]
    gen_thresh = defaults["GEN_THRESH"]
    crossover_rate = defaults["crossover_rate"]
    mutation_rate = defaults["mutation_rate"]

    # Initialize variables
    no_improvement_counter = 0
    best_overall_fitness = float("inf")

    # Step 1: Initialize population
    population = initialize_population(nb_cities, pop_size, distance_matrix)

    # Step 2: Iterate through generations
    for gen in range(gen_thresh):
        # Sort by fitness
        population.sort()
        current_best_fitness = population[0].fitness

        # Log generation details
        print(f"\nGeneration {gen + 1}")
        # print(f"Temperature: {temperature:.2f}")
        print(f"Best Fitness: {current_best_fitness}")
        print("Top Individual:", population[0].gnome)

        # Check for improvements
        # if current_best_fitness < best_overall_fitness:
        #     best_overall_fitness = current_best_fitness
        #     no_improvement_counter = 0
        # else:
        #     no_improvement_counter += 1

        # Step 3: Generate new population
        new_population = [population[0]]  # Elitism

        # Crossover
        while len(new_population) < pop_size:
            parent1 = select_parent(population)
            parent2 = select_parent(population)

            # Keep parents if crossover is not applied
            if random.random() < crossover_rate:
                child = apply_crossover(parent1, parent2, distance_matrix)
                print("Child:", type(child))
                new_population.append(child)
            else:
                new_population.append(parent1)
                new_population.append(parent2)

        # Mutation
        while len(new_population) <= pop_size:
            parent = select_parent(population)
            # mutated_individual = apply_mutation(parent, mutation_rate, distance_matrix)
            if random.random() < mutation_rate:
                mutated_gnome = mutated_gene(parent.gnome, len(distance_matrix))
                parent.gnome = mutated_gnome
                parent.fitness = cal_fitness(mutated_gnome, distance_matrix)
            # Add the parent (mutated or not) to the new population
            new_population.append(parent)

        # # Dynamic termination
        # if no_improvement_counter >= 50:  # Stop if no improvement in 50 generations
        #     break

        # Step 4: Refine population
        # population = refine_population(new_population, distance_matrix)
        print(new_population[0])
        population = new_population

        # Step 5: Cool down temperature
        # temperature *= cooling_rate

        # check if len tour is equal to nb_cities
        print("Length of tour:", len(population[0].gnome))

    # Best solution in the final population
    best_individual = min(population, key=lambda x: x.fitness)
    cost = best_individual.fitness
    seq = [x+1 for x in best_individual.gnome]
    print("\nBest sequence:", seq)
    print("Cost:", cost)
    return cost, seq

def objective(trial):
    params = {
        "POP_SIZE": trial.suggest_int("POP_SIZE", 50, 100, step=10),
        "GEN_THRESH": trial.suggest_int("GEN_THRESH", 10, 100, step=10),
        "crossover_rate": trial.suggest_float("crossover_rate", 0.5, 1.0),
        "mutation_rate": trial.suggest_float("mutation_rate", 0.05, 0.3),
    }
    start_time = time.time()
    print(f"Trial {trial.number}: Hyperparameters = {params}")
    best_individual = genetic(distance_matrix, hyperparams=params)
    print(f"Trial {trial.number}: Fitness = {best_individual.fitness}\n")
    time_to_converge = time.time() - start_time

    fitness_with_time_penalty = best_individual.fitness + (time_to_converge * 0.1)
    return fitness_with_time_penalty


if __name__ == "__main__":
    # SEED = 42  # Set a fixed seed for reproducibility
    # random.seed(SEED)
    # np.random.seed(SEED)

    data = "eil101"  # Specify your dataset
    problem = tsplib95.load(f'../data/ALL_tsp/{data}.tsp')
    distance_matrix = create_distance_matrix(problem)

    hyperparams = {
        "POP_SIZE": 10,
        "GEN_THRESH": 10,
        "cooling_rate": 0.9,
        "crossover_rate": 0.7,
        "mutation_rate": 0.05,
    }

    solution = genetic(distance_matrix, hyperparams)

    opt_cost = get_optimal_cost(opt_sol.data, data)
    print("\nOptimal cost:", opt_cost)

    print("\nRelative error:", (solution[0] - opt_cost) / opt_cost)

    # Run optimization
    # study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    # study.optimize(objective, n_trials=20)
    #
    # print("\nBest Parameters:", study.best_params)
    # print("Best Fitness:", study.best_value)
    #
    # optuna.visualization.plot_optimization_history(study).show()
    # optuna.visualization.plot_param_importances(study).show()
