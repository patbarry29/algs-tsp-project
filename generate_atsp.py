import numpy as np

def generate_atsp(n, dim_size=100):
    # Generate a random asymmetric distance matrix
    distance_matrix = np.random.randint(1, dim_size, size=(n, n))
    np.fill_diagonal(distance_matrix, np.max(distance_matrix)*10000)  # Ensure no self-loops (distance to self is 0)

    # Save to a file in ATSP format
    save_atsp_file(distance_matrix)

def save_atsp_file(distance_matrix, filename="random_atsp.atsp"):
    n = distance_matrix.shape[0]
    with open(filename, "w") as f:
        # Write the header
        f.write("NAME : Random_ATSP_Instance\n")
        f.write("TYPE : ATSP\n")
        f.write(f"DIMENSION : {n}\n")
        f.write("EDGE_WEIGHT_TYPE : EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT : FULL_MATRIX\n")
        f.write("EDGE_WEIGHT_SECTION\n")

        # Write the distance matrix
        for row in distance_matrix:
            f.write(" ".join(map(str, row)) + "\n")

        # End of file
        f.write("EOF\n")

# Example usage
generate_atsp(11, dim_size=100)
