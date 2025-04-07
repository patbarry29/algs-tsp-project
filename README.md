5-person group project to build and test several algorithms to solve the Travelling Salesman Problem using Python.

Each person contributed 2 algorithms.
- [@patbarry29](https://github.com/patbarry29): Brute Force, and Ant-Colony Optimisation
- [@dinhhb](https://github.com/dinhhb): Greedy, and Genetic Algorithms
- [@mahimahari](https://github.com/mahimahari): Branch and Bound (B+B), and B+B with Edge Selection and Matrix Reduction
- [@ChelsyMena](https://github.com/ChelsyMena): Dynamic Programming, and Randomised Programming (Monte Carlo method)
- [@vakulenko-serhii
](https://github.com/vakulenko-serhii): Minimum Spanning Tree, and Lin-Kernighan

# Guidelines for using Problem Generator and Distance Matrix
If you want to use the random generator for TSP or ATSP inside your algorithm, you have to import the functions awkwardly.
First do all your other imports (like ones from inside your own directory) and then run this:

```python
# all other imports up here
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from create_distance_matrix import create_distance_matrix
from generate_atsp import generate_atsp
```

Then once you have imported the problem generator, you can manipulate 3 parameters.
- *n*: the size of the problem (number of cities).
- *dim_size*: the size of the space the problems will be in.
  - e.g. `dim_size=10`, all points will have an x-value in the range 0-10 and same for y-value.
- *sparsity*: defines the level of connectedness of the graph.
  - accepts values in the range 0-1.
