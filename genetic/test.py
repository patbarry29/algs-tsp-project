import numpy as np
import tsplib95
from sko.GA import GA_TSP
from utils.create_distance_matrix import create_distance_matrix
from data.opt_cost import tsp as opt_sol
import matplotlib.pyplot as plt
from utils.get_opt_cost import get_optimal_cost

data = "a280"  # Specify your dataset
problem = tsplib95.load(f'../data/ALL_tsp/{data}.tsp')
distance_matrix = create_distance_matrix(problem)

num_points = len(distance_matrix)
points_coordinate = np.random.rand(num_points, 2)  # generate coordinate of points


def cal_total_distance(routine):
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

def test_GA():

    ga_tsp = GA_TSP(func=cal_total_distance, n_dim=len(distance_matrix), size_pop=200, max_iter=5000, prob_mut=0.3)
    best_points, best_distance = ga_tsp.run()
    print('best_points:', best_points, 'best_distance:', best_distance)

    opt_cost = get_optimal_cost(opt_sol.data, data)
    print("\nOptimal cost:", opt_cost)

    # print("\nRelative error:", (solution[0] - opt_cost) / opt_cost)

    fig, ax = plt.subplots(1, 2)
    best_points_ = np.concatenate([best_points, [best_points[0]]])
    best_points_coordinate = points_coordinate[best_points_, :]
    ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
    ax[1].plot(ga_tsp.generation_best_Y)
    plt.show()


test_GA()