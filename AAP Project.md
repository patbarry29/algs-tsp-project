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
- [ ] Dynamic Programming Approach
	- [x] Read the paper: Bouman et al. uploaded on claroline
	- [x] Read the wikipedia page of the Held-Karp algorithm: https://en.wikipedia.org/wiki/Held%E2%80%93Karp_algorithm.
	- [x] Get cost of chosen path
	- [x] Get chosen path
	- [ ] Test with other type of problems
- [ ] Randomized Approach
- [ ] Graph Generator
- [ ] GUI

## Links 
[TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/index.html)

## Notes
- There must special considerations for the unlinked nodes in that kind of problem. Should we initialize the matrix at inf to make a clearer distinction?