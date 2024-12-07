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

# Adapted from https://www.geeksforgeeks.org/traveling-salesman-problem-using-genetic-algorithm/

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
    shuffle(gnome[1:])  # Shuffle only non-starting cities
    gnome.append(gnome[0])  # Return to the starting city
    return gnome


def mutated_gene(gnome, nb_cities):
    """Introduces randomness by swapping two cities in the GNOME
    (excluding the starting/ending city)."""
    gnome = gnome[:-1]  # Exclude the last element (starting city)
    while True:
        r = rand_num(1, nb_cities)
        r1 = rand_num(1, nb_cities)
        if r != r1:
            gnome[r], gnome[r1] = gnome[r1], gnome[r]
            break
    gnome.append(gnome[0])  # Ensure the GNOME is cyclic
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


def cooldown(temp):
    """Reduces the temperature."""
    return temp * 0.95  # Reduce temperature by 5% per generation


def two_opt(gnome, distance_matrix):
    """
    Applies the 2-opt local search to refine a GNOME (tour).
    """""
    # Initialize the best tour as the current GNOME
    best = gnome[:]
    # Calculate the cost of the current best tour
    best_cost = cal_fitness(best, distance_matrix)
    # Set a flag to track if improvements are made
    improved = True

    # Continue searching for improvements until no changes are found
    while improved:
        improved = False  # Reset the improvement flag for this iteration

        # Loop over all possible pairs of edges in the tour
        for i in range(1, len(best) - 2):  # Start from the second city (index 1)
            for j in range(i + 1, len(best) - 1):  # Ensure j > i to avoid duplicate checks
                if j - i == 1:  # Skip adjacent nodes, as swapping adjacent edges is unnecessary
                    continue

                # Create a new GNOME by reversing the segment between i and j
                new_gnome = best[:]  # Copy the current best tour
                new_gnome[i:j] = best[j - 1:i - 1:-1]  # Reverse the segment between indices i and j

                # Calculate the cost of the new GNOME
                new_cost = cal_fitness(new_gnome, distance_matrix)

                # If the new GNOME has a lower cost, update the best tour
                if new_cost < best_cost:
                    best = new_gnome[:]  # Update the best tour
                    best_cost = new_cost  # Update the best cost
                    improved = True  # Set the improvement flag to True

    # Return the refined tour after no further improvements are found
    return best


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
    """Selects a parent for reproduction."""
    return population[rand_num(0, len(population))]


def apply_crossover(parent1, parent2, distance_matrix):
    """Applies crossover to generate offspring (currently using mutation as a placeholder)."""
    child_gnome = mutated_gene(parent1.gnome, len(distance_matrix))  # Placeholder for proper crossover
    child_fitness = cal_fitness(child_gnome, distance_matrix)
    child = Individual()
    child.gnome = child_gnome
    child.fitness = child_fitness
    return child


def apply_mutation(individual, mutation_rate, distance_matrix):
    """Applies mutation to an individual with a given probability."""
    if randint(0, 100) / 100.0 < mutation_rate:
        mutated_gnome = mutated_gene(individual.gnome, len(distance_matrix))
        individual.gnome = mutated_gnome
        individual.fitness = cal_fitness(mutated_gnome, distance_matrix)
    return individual


def refine_population(population, distance_matrix):
    """Applies 2-opt refinement to improve solutions in the population."""
    for individual in population:
        refined_gnome = two_opt(individual.gnome, distance_matrix)
        individual.gnome = refined_gnome
        individual.fitness = cal_fitness(refined_gnome, distance_matrix)
    return population


def genetic(distance_matrix, hyperparams=None):
    """Main TSP Genetic Algorithm logic."""

    # Default hyperparameters
    defaults = {
        "POP_SIZE": 50,
        "GEN_THRESH": 100,
        "temperature": 10000,
        "cooling_rate": 0.95,
        "crossover_rate": 0.8,
        "mutation_rate": 0.2,
    }
    if hyperparams:
        defaults.update(hyperparams)

    nb_cities = len(distance_matrix)
    pop_size = defaults["POP_SIZE"]
    gen_thresh = defaults["GEN_THRESH"]
    temperature = defaults["temperature"]
    cooling_rate = defaults["cooling_rate"]
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
        # print(f"Best Fitness: {current_best_fitness}")
        # print("Top Individual:", population[0].gnome)

        # Check for improvements
        if current_best_fitness < best_overall_fitness:
            best_overall_fitness = current_best_fitness
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1

        # Step 3: Generate new population
        new_population = [population[0]]  # Elitism

        # Crossover
        num_crossover = int(crossover_rate * pop_size)
        while len(new_population) < num_crossover:
            parent1 = select_parent(population)
            parent2 = select_parent(population)
            child = apply_crossover(parent1, parent2, distance_matrix)
            new_population.append(child)

        # Mutation
        while len(new_population) < pop_size:
            parent = select_parent(population)
            mutated_individual = apply_mutation(parent, mutation_rate, distance_matrix)
            new_population.append(mutated_individual)

        # Step 4: Refine population
        population = refine_population(new_population, distance_matrix)

        # Step 5: Cool down temperature
        temperature *= cooling_rate

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
    SEED = 42  # Set a fixed seed for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)

    data = "att48"  # Specify your dataset
    problem = tsplib95.load(f'../data/ALL_tsp/{data}.tsp')
    distance_matrix = create_distance_matrix(problem)

    hyperparams = {
        "POP_SIZE": 10,
        "GEN_THRESH": 20,
        "cooling_rate": 0.9,
        "crossover_rate": 0.7,
        "mutation_rate": 0.3,
    }

    solution = genetic(distance_matrix, hyperparams)

    opt_cost = get_optimal_cost(opt_sol.data, data)
    print("\nOptimal cost:", opt_cost)

    # Run optimization
    # study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    # study.optimize(objective, n_trials=20)
    #
    # print("\nBest Parameters:", study.best_params)
    # print("Best Fitness:", study.best_value)
    #
    # optuna.visualization.plot_optimization_history(study).show()
    # optuna.visualization.plot_param_importances(study).show()
