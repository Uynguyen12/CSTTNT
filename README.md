# CSC14003 – Fundamentals of Artificial Intelligence
## Team Project 01: Search & Nature-Inspired Algorithms
### Group 9 | HCMUS – Faculty of Information Technology | 2026

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Algorithms Implemented](#algorithms-implemented)
- [Problems & Test Cases](#problems--test-cases)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Running Benchmarks](#running-benchmarks)
- [Results](#results)
- [Dependencies](#dependencies)
- [Team Members](#team-members)

---

## Project Overview

This project implements, analyzes, and compares **classical graph search algorithms** and **nature-inspired optimization algorithms** from scratch using only NumPy and basic Python.

Four optimization problems are addressed:
- **Graph Search** – shortest path on weighted undirected graphs
- **Travelling Salesman Problem (TSP)** – minimize total tour length
- **0/1 Knapsack** – maximize value under weight constraint
- **Continuous Benchmark Optimization** – minimize standard benchmark functions

---

## Project Structure

```
CSTTNT/
├── algorithms/
│   ├── classical/
│   │   ├── uninformed.py          # BFS, DFS, UCS
│   │   ├── informed.py            # Greedy Best-First, A*
│   │   └── local_search.py        # Hill Climbing, Simulated Annealing (graph)
│   └── nature_inspired/
│       ├── evolution_based.py     # GA, DE
│       ├── physics_based.py       # SA (continuous), GSA, HS
│       ├── swarm_based.py         # PSO, ACO/ACOR, ABC, FA, CS
│       └── human_based.py         # TLBO (bonus)
│
├── utils/
│   ├── graph.py                   # Graph data structure
│   ├── tsp_utils.py               # TSP distance matrix, tour length
│   └── knapsack_utils.py          # Knapsack encoding, penalty
│
├── test_cases/
│   ├── graph/                     # tc01_trivial_chain.txt ... tc12_expert_100nodes.txt
│   ├── tsp/                       # tc01_trivial_5cities.txt ... tc08_expert_100cities.txt
│   ├── knapsack/                  # tc01_trivial_5items.txt ... tc07_expert_100items.txt
│   └── continuous/
│       └── continuous_testcases.txt
│
├── results/                       # Auto-generated CSV outputs
│   ├── graph_results.csv
│   ├── tsp_results.csv
│   ├── knapsack_results.csv
│   └── continuous_results.csv
│
├── plots/                         # Auto-generated PNG visualizations
│   ├── benchmark_functions.png
│   ├── convergence_curves.png
│   ├── graph_scalability.png
│   ├── knapsack_comparison.png
│   ├── robustness_boxplot.png
│   ├── hypothesis_testing_heatmap.png
│   └── exploration_exploitation.png
│
├── benchmark_experiments.ipynb    # Main benchmark notebook
├── requirements.txt
└── README.md
```

---

## Algorithms Implemented

### Classical Search (Graph)

| Category | Algorithm | Key Property |
|---|---|---|
| Uninformed | BFS | Optimal hops, complete |
| Uninformed | DFS | Memory efficient, not optimal |
| Uninformed | UCS | Optimal cost, complete |
| Informed | Greedy Best-First | Fast, not guaranteed optimal |
| Informed | A* | Optimal + efficient (admissible heuristic) |
| Local Search | Hill Climbing | Fast, susceptible to local optima |
| Local Search | Simulated Annealing | Escapes local optima via probabilistic acceptance |

### Nature-Inspired Algorithms

| Category | Algorithm | Applied To |
|---|---|---|
| Evolution | Genetic Algorithm (GA) | TSP, Knapsack, Continuous |
| Evolution | Differential Evolution (DE) | TSP, Knapsack, Continuous |
| Physics | Simulated Annealing (SA) | Continuous |
| Physics | Gravitational Search Algorithm (GSA) | Continuous |
| Physics | Harmony Search (HS) | Continuous |
| Swarm | Particle Swarm Optimization (PSO) | TSP, Knapsack, Continuous |
| Swarm | Ant Colony Optimization (ACOR) | Continuous |
| Swarm | Artificial Bee Colony (ABC) | Knapsack, Continuous |
| Swarm | Firefly Algorithm (FA) | TSP, Continuous |
| Swarm | Cuckoo Search (CS) | TSP, Knapsack, Continuous |
| Human | TLBO *(bonus)* | Knapsack, Continuous |

---

## Problems & Test Cases

### Graph Search — 12 test cases (TC01–TC12)
- Size: 3 to 100 nodes
- 7 algorithms evaluated on every test case
- Metrics: path cost, path length, nodes expanded, execution time

### TSP — 8 test cases (TC01–TC08)
- Size: 5 to 100 cities, 2D Euclidean distance
- Algorithms: GA, DE, PSO, CS, FA
- Encoding: continuous relaxation with rank-sort decoding

### Knapsack — 7 test cases (TC01–TC07)
- Size: 5 to 100 items
- Algorithms: GA, DE, PSO, ABC, CS, TLBO + Greedy baseline
- Encoding: continuous vector, decode via threshold x[i] > 0.5
- Penalty: `max(0, over_weight) × max(value) × 2`

### Continuous Optimization — 24 test cases (TC01–TC24)
- Functions: Sphere, Rastrigin, Rosenbrock, Griewank, Ackley
- Dimensions: D = 2, 5, 10, 20, 30, 50, 100
- 11 algorithms evaluated
- Metrics: best fitness, mean ± std over 5 runs, convergence curve

---

## Setup & Installation

### Requirements
- Python 3.10+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Jupyter Notebook

### Install dependencies

```bash
pip install -r requirements.txt
```

### requirements.txt
```
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
scipy>=1.10.0
```

> ⚠️ **Note:** All optimization algorithms are implemented using NumPy only — no scipy.optimize, scikit-learn, or other high-level optimization libraries are used in the algorithm implementations.

---

## Usage

### Run a single algorithm on a graph test case

```python
from algorithms.classical.informed import AStarSearch
from utils.graph import Graph

# Load graph
g, start, goal = load_graph("test_cases/graph/tc07_medium_dense.txt")

# Run A*
algo = AStarSearch(heuristic="euclidean")
result = algo.search(g, start, goal)

print(f"Path: {result.path}")
print(f"Cost: {result.cost:.3f}")
print(f"Nodes explored: {result.nodes_explored}")
```

### Run a nature-inspired algorithm on Knapsack

```python
from algorithms.nature_inspired.human_based import TeachingLearningBasedOptimization
import numpy as np

weights = np.array([...])
values  = np.array([...])
capacity = 300

def kp_objective(x):
    items = (x > 0.5).astype(int)
    penalty = max(0, np.dot(items, weights) - capacity) * max(values) * 2
    return -np.dot(items, values) + penalty

problem = {
    "objective_func": kp_objective,
    "dimensions": len(weights),
    "lower_bound": 0.0,
    "upper_bound": 1.0
}

tlbo = TeachingLearningBasedOptimization(population_size=50, max_iterations=300, seed=42)
result = tlbo.optimize(problem)

items_selected = (result.solution > 0.5).astype(int)
total_value  = np.dot(items_selected, values)
total_weight = np.dot(items_selected, weights)
print(f"Value: {total_value} | Weight: {total_weight} | Feasible: {total_weight <= capacity}")
```

### Run continuous optimization

```python
from algorithms.nature_inspired.swarm_based import ParticleSwarmOptimization
import numpy as np

def rastrigin(x):
    return float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))

problem = {
    "objective_func": rastrigin,
    "dimensions": 10,
    "lower_bound": -5.12,
    "upper_bound": 5.12
}

pso = ParticleSwarmOptimization(population_size=30, max_iterations=300, seed=42)
result = pso.optimize(problem)
print(f"Best fitness: {result.best_fitness:.6e}")
```

---

## Running Benchmarks

All experiments are organized in `benchmark_experiments.ipynb`.

### Open notebook

```bash
jupyter notebook benchmark_experiments.ipynb
```

### Run all experiments

Select **Kernel → Restart & Run All**.

The notebook will automatically:
1. Load all test cases from `test_cases/`
2. Run all algorithms on all problems
3. Export results to `results/*.csv`
4. Generate all plots to `plots/*.png`

### Output files

| File | Description |
|---|---|
| `results/graph_results.csv` | Path cost, nodes explored, time per algorithm per TC |
| `results/tsp_results.csv` | Best tour cost, time per algorithm per TC |
| `results/knapsack_results.csv` | Total value, feasibility, time per algorithm per TC |
| `results/continuous_results.csv` | Best, mean, std, time per algorithm per TC |

---

## Results Summary

### Graph Search
- **Best overall:** A* — optimal cost + fewest nodes expanded
- **Fastest:** Hill Climbing — but unreliable on complex topologies
- **Worst cost quality:** DFS — finds a path but not optimal

### Continuous Optimization (D=10)
- **Unimodal (Sphere, Ackley):** TLBO dominates, PSO close second
- **Multimodal (Rastrigin):** ABC best, most others struggle
- **Most stable:** GA, DE, PSO, FA (low variance over 20 runs)
- **Least stable:** SA (very high variance), CS (frequent outliers)

### TSP
- **Best:** PSO, GA, ABC in current setup
- **Note:** All algorithms use rank-sort encoding — results would improve significantly with direct permutation encoding

### Knapsack
- **Best metaheuristic:** TLBO, then ABC
- **Greedy baseline** remains competitive under current budget (pop=50, iter=300)
- **Ranking:** TLBO > GA ≈ DE > PSO > ABC > CS

---

## Notes & Limitations

- All results are based on fixed budget: `population=50, max_iter=300, n_runs=3` (TSP/Knapsack) and `population=30, max_iter=300, n_runs=5` (Continuous).
- DE underperforms relative to theoretical expectations — hyperparameters (F, CR) have not been tuned per problem.
- TSP encoding (rank-sort) introduces structural mismatch for PSO and FA — direct permutation encoding would improve results.
- Hypothesis testing currently excludes TLBO — should be added in future iterations.
- For statistically reliable comparisons, `n_runs ≥ 20–30` is recommended.

---

## Dependencies

```
numpy          — all algorithm implementations
pandas         — results aggregation and CSV export
matplotlib     — convergence curves, scalability charts, tour visualizations
seaborn        — heatmaps, boxplots
jupyter        — benchmark notebook
scipy          — hypothesis testing (Mann-Whitney U, Wilcoxon)
```

---

## References

[1] Dorigo, M., Birattari, M., & Stutzle, T. (2007). Ant colony optimization. *IEEE Computational Intelligence Magazine, 1*(4), 28–39.

[2] Wang, D., Tan, D., & Liu, L. (2018). Particle swarm optimization algorithm: an overview. *Soft Computing, 22*(2), 387–408.

[3] Karaboga, D., & Basturk, B. (2007). A powerful and efficient algorithm for numerical function optimization: artificial bee colony (ABC) algorithm. *Journal of Global Optimization, 39*(3), 459–471.

[4] Yang, X. S., & He, X. (2013). Firefly algorithm: recent advances and applications. *International Journal of Swarm Intelligence, 1*(1), 36–50.

[5] Yang, X. S., & Deb, S. (2014). Cuckoo search: recent advances and applications. *Neural Computing and Applications, 24*(1), 169–174.