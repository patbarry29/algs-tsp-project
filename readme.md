[[00 Main - Advanced Algorithmics and Programming]]

# Project 

Traveling Salesman Problem
All the algorithms must be able to output the solution 
- The ordered sequence of nodes to visit
- The cost of the proposed solution

The running time of the algorithms and the quality of the solutions given (in terms of optimality) have to be evaluated on problems of different sizes and structure (i.e. with different number of cities, different distances between cities, different underlying graphs, . . . )

**Optional**
Graphical user interface to illustrate the behavior of the algorithm(s)
Random generator able to create automatically some TSP problems. It can take into account some options like the level of sparsity of the problem (i.e. level of connectivity), a range over possible edge weights, a distribution for generating the weights, ...)

**TA**
erick.gary.gomez.soto@univ-st-etienne.fr

## Tasks 

#todo/Masters/AAP
- [x] Dynamic Programming Approach
	- [x] Read the paper: Bouman et al. uploaded on claroline
	- [x] Read the wikipedia page of the Held-Karp algorithm: https://en.wikipedia.org/wiki/Held%E2%80%93Karp_algorithm.
	- [x] Get cost of chosen path
	- [x] Get chosen path
	- [x] Test with other type of problems
- [ ] Randomized Approach
	- [ ] Understand the papers maybe
	- [ ] MST
	- [ ] Perfect matching
	- [ ] Graph combination
	- [ ] Eulerian Tour
	- [ ] TSP
- [x] Graph Generator
- [ ] GUI

## Links 
[TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/index.html)

## Notes
- There must special considerations for the unlinked nodes in that kind of problem. Should we initialize the matrix at inf to make a clearer distinction?
- We can use the 1-tree minimum spanning tree to approximate or lower bound the optimal tsp cost for the ones we generate randomly

Randomised: 
Get the MST, a graph that had all nodes joined with no cycles and it's min weight. -- prim's algo
	- random vertex vs the others. Take the min distance and remove the one you choose from the others set
	- check all distances from the ones you've been in to the ones you havent and pick the min, until you have gone to all of them

A MST has edges that are only connected to one or three vertex instead of two, in a TSP ever node gets two connections. Pair up those odd-connection nodes.

| ![image](<assets/AAP Project/file-20241119135403511.png>) | ![image](<assets/AAP Project/file-20241119135444193.png>) |
| --------------------------------------------------------- | --------------------------------------------------------- |

perfect matching of the vertices -- get the best pairs, the less costly pairs.
- [?] What do you do if the odd nodes... are odd? which one do you ignore?

Then you add the two graphs, the MST + the perfect matching

Then make an eulerian tour of that graph, 
	- remove duplicate edges
Then you make that into a TSP tour by removing any doubling down there is. 
	- any path that goes to every node of the remaining graph, skipping any you've already been in.
The cost will be close to the optimum, within 50% of it. 

![](<assets/AAP Project/file-20241119132402233.png>)

Improve after this
- random swapping
- 2-opt
- 3-opt
- k-opt

Library to compare against maybe 
networkx==2.8.8