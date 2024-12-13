parameters to tune
### Parameters to Tune

#### Number of Ants

### Alpha and Beta

| Alpha | Beta | Average Cost |
|-------|------|--------------|
| 0.5   | 1.0  | 1057.2046    |
|    | 1.5  | 830.0618     |
|    | 2.0  | 736.0861     |
|    | 2.5  | 675.5328     |
|    | 3.0  | 652.4948     |
|    | 3.5  | 631.7263     |
|    | 4.0  | 626.3128     |
|    | 4.5  | 620.1459     |
| 1.0   | 1.0  | 715.5176     |
|    | 1.5  | 648.3922     |
|    | 2.0  | 621.8419     |
|    | 2.5  | 614.8406     |
|    | 3.0  | 604.3485     |
|    | 3.5  | 604.3847     |
|    | 4.0  | 599.9771     |
|    | 4.5  | 604.0828     |
| 1.5   | 1.0  | 651.0331     |
|    | 1.5  | 630.3946     |
|    | 2.0  | 616.6096     |
|    | 2.5  | 615.5943     |
|    | 3.0  | 608.3375     |
|    | 3.5  | 610.2501     |
|    | 4.0  | 609.8416     |
|    | 4.5  | 611.5482     |
| 2.0   | 1.0  | 674.3441     |
|    | 1.5  | 640.2313     |
|    | 2.0  | 632.5847     |
|    | 2.5  | 622.9324     |
|    | 3.0  | 622.5061     |
|    | 3.5  | 616.4442     |
|    | 4.0  | 617.4211     |
|    | 4.5  | 614.5889     |

#### Number of Iterations

- `num = 350` is the best value. The higher the number of iterations, the better.
    - `num = 10` vs `num = 350`
        - `num = 10` won 1/100 times
        - `num = 350` won 91/100 times
    - `num = 50` vs `num = 300`
        - `num = 50` won 12/100 times
        - `num = 300` won 73/100 times
    - `num = 100` vs `num = 250`
        - `num = 100` won 23/100 times
        - `num = 250` won 64/100 times
    - `num = 150` vs `num = 200`
        - `num = 150` won 40/100 times
        - `num = 200` won 43/100 times
    - `num = 350` vs `num = 250`
        - `num = 350` won 52/100 times
        - `num = 250` won 33/100 times

#### Initial Pheromone Level

- The value doesn't seem to have much effect on the result.

#### Pheromone Evaporation Rate

- The value doesn't seem to have much effect on the result.

#### Number of Ants

- The value doesn't seem to have much effect on the result.
- But it does have an effect on the time.
    - So we will go for a smaller number of ants (around 50).