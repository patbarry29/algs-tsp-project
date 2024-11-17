import numpy as np

def generate_random_weights(distance_matrix, weight_range=(1, 100)):
    """
    Replace non-zero distances with random weights within the range.
    weight_range: (min_weight, max_weight).
    """
    random_weights = np.random.randint(weight_range[0], weight_range[1], size=distance_matrix.shape)
    random_weighted_matrix = np.where(distance_matrix > 0, random_weights, 0)
    return random_weighted_matrix

def generate_weight_distribution(distance_matrix, distribution="uniform", params={}):
    """
    Replace non-zero distances with weights based on a chosen distribution.
    distribution: 'uniform', 'normal', or 'exponential'.
    params: Parameters for the distribution.
    """
    if distribution == "uniform":
        weights = np.random.uniform(params.get("low", 1), params.get("high", 100), size=distance_matrix.shape)
    elif distribution == "normal":
        weights = np.random.normal(params.get("mean", 50), params.get("std", 10), size=distance_matrix.shape)
    elif distribution == "exponential":
        weights = np.random.exponential(params.get("scale", 10), size=distance_matrix.shape)
    else:
        raise ValueError("Unsupported distribution")

    weighted_matrix = np.where(distance_matrix > 0, weights, 0)
    return weighted_matrix
