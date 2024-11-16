import numpy as np

from visualise_problem import visualise

def generate_tsp(n,dim_size=100):
  cities = np.random.uniform(0, dim_size, size=(n, 2))
  save_tsp_file(cities)
  return cities

def save_tsp_file(cities, filename="random_tsp.tsp"):
  with open(filename, "w") as f:
    # Write the header
    f.write("NAME : Random_TSP_Instance\n")
    f.write("TYPE : TSP\n")
    f.write(f"DIMENSION : {len(cities)}\n")
    f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
    f.write("NODE_COORD_SECTION\n")

    # Write the city coordinates
    for i, (x, y) in enumerate(cities, start=1):
        f.write(f"{i} {x:.6f} {y:.6f}\n")

    # End of file
    f.write("EOF\n")

cities = generate_tsp(10, 10)

visualise(cities)