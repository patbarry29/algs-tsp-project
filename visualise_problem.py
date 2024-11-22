import matplotlib.pyplot as plt

def visualise(problem):
    x_coords, y_coords = zip(*problem)
    plt.scatter(x_coords, y_coords)
    for i, (x, y) in enumerate(problem):
        plt.text(x, y, str(i), fontsize=12)
    plt.title(f"Random TSP Instance of Size {len(problem)}")
    plt.show()
