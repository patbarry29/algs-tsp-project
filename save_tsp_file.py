def save_matrix_file(cities=None, matrix=None, filename="random_tsp.tsp", type="TSP", is_symmetric=True):
    """
    Saves a matrix (distance or cost matrix) to a TSP or ATSP format file.

    Args:
      matrix: The matrix (numpy array) to save.
      filename: The name of the file to write.
      name: The name of the problem instance (e.g., "Random_TSP_Instance").
      type: The type of problem ("TSP" or "ATSP").
      is_symmetric: True if the matrix is symmetric (TSP), False otherwise (ATSP).
    """
    with open(filename, "w") as f:
        # Write the header
        f.write(f"NAME : Random_{type}_Instance\n")
        f.write(f"TYPE : {type}\n")
        dimension = matrix.shape[0] if matrix is not None else len(cities)
        f.write(f"DIMENSION : {dimension}\n")

        if is_symmetric:
            if matrix is not None:
                f.write("EDGE_WEIGHT_TYPE : EXPLICIT\n")
                f.write("EDGE_WEIGHT_FORMAT : FULL_MATRIX\n")
                f.write("NODE_COORD_TYPE : NO_COORDS\n")
                f.write("DISPLAY_DATA_TYPE: TWOD_DISPLAY\n")
            else:
                f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        else: # ATSP
            f.write("EDGE_WEIGHT_TYPE : EXPLICIT\n")
            f.write("EDGE_WEIGHT_FORMAT : FULL_MATRIX\n")

        # Write the matrix section (coordinates for TSP or matrix for ATSP)
        if is_symmetric and matrix is None:  # TSP with Euclidean coordinates
            f.write("NODE_COORD_SECTION\n")
            for i, (x, y) in enumerate(cities, start=1):
              f.write(f"{i} {x:.6f} {y:.6f}\n")
        else:  # TSP with explicit matrix or ATSP matrix
            f.write("EDGE_WEIGHT_SECTION\n")
            for row in matrix:
                f.write(" ".join(f"{value:.6f}" if isinstance(value, float) else str(value) for value in row) + "\n")

        # End of file
        f.write("EOF\n")