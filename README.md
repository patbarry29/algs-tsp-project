# Guidelines for using Problem Generator and Distance Matrix
If you want to use the random generator for TSP or ATSP inside your algorithm, you have to import the functions awkwardly.
First do all your other imports (like ones from inside your own directory) and then run this:

```python
# all other imports up here

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

# Our chosen TSP Problems (irrelevant now?)
Choose 4 TSP problems (2 medium or 1 extra large?)
- Small:  burma14.tsp
- Medium: berlin52.tsp
- Large:  gr666.tsp
- XL:     pr2392.tsp

Choose 3 ATSP problems
- Small:  br17.atsp
- Medium: ftv64.atsp
- Large:  rbg358.atsp

Choose 1 HCP problems
- Large: alb3000a.hcp

Choose 2-3 SOP problems
- Small:  esc07.sop
- Medium: rbg048a.sop
- Large:  rbg341.sop

Choose 2-3 VRP problems
- Small:  eil13.vrp
- Medium: eil51.vrp
- Large:  gil262.vrp
