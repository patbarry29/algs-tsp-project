# Usage:
# e.g. to get the optimal cost for the bayg29 (tsp) instance:
# from data.opt_cost import tsp as opt_sol
# optimal_cost = get_optimal_cost(opt_sol.data, "bayg29")

def get_optimal_cost(data, instance):
    # Create a dictionary for quick lookup
    optimal_cost_dict = dict(zip(data["Instance"], data["OptimalCost"]))

    if instance not in optimal_cost_dict:
        print(f"Instance {instance} not found in the optimal cost data !!!")
        return None

    optimal_cost = optimal_cost_dict[instance]
    return optimal_cost