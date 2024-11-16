import numpy as np

def generate_tsp(n,dim_size=100):
  cities = np.random.uniform(0, dim_size, size=(n, 2))
  save_tsp_file(cities)

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

def visualise(problem):
  import matplotlib.pyplot as plt
  x_coords, y_coords = zip(*problem)
  plt.scatter(x_coords, y_coords)
  for i, (x, y) in enumerate(problem):
      plt.text(x, y, str(i), fontsize=12)
  plt.title(f"Random TSP Instance of Size {len(problem)}")
  plt.show()


generate_tsp(8, 100)

# visualise(cities)