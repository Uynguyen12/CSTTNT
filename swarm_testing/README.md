# Swarm-Based Algorithms - Test Cases & Visualization

## 📋 Tổng Quan

Package này cung cấp test cases và visualization đầy đủ cho các thuật toán swarm-based theo yêu cầu đồ án. Bao gồm:

### 🔬 Algorithms Được Test
- **PSO** - Particle Swarm Optimization
- **ACO** - Ant Colony Optimization  
- **ABC** - Artificial Bee Colony
- **FA** - Firefly Algorithm
- **CS** - Cuckoo Search

### 📊 Test Cases

#### Continuous Optimization (Bài toán liên tục)
- **Unimodal Functions** (kiểm tra convergence speed):
  - Sphere Function (5D, 10D, 20D)
  - Rosenbrock Function (5D, 10D)

- **Multimodal Functions** (kiểm tra escape local minima):
  - Rastrigin Function (5D, 10D, 20D) - highly multimodal
  - Ackley Function (5D, 10D) - many local minima
  - Griewank Function (10D) - regularly distributed minima
  - Schwefel Function (5D, 10D) - highly deceptive
  - Levy Function (10D) - many local minima

- **2D Visualization Functions**:
  - Beale, Booth, Matyas

#### Discrete Optimization (Bài toán rời rạc lớn)
- **TSP** - Traveling Salesman Problem (30, 50, 75, 100 cities)
  - Với constraints: time windows, service times
  - Multi-objective: distance cost + time cost + penalties
  
- **Knapsack Problem** (50, 100, 150, 200 items)
  - Capacity constraints
  - Value-weight correlations
  
- **Graph Coloring** (30, 50, 75, 100 vertices)
  - Minimize number of colors
  - No adjacent vertices same color

---

## 📁 File Structure

```
swarm_testing/
│
├── test_cases_discrete.py          # Discrete problem generators
│   ├── TSPInstance
│   ├── KnapsackInstance
│   └── GraphColoringInstance
│
├── test_cases_continuous.py        # Continuous benchmark functions
│   ├── Sphere, Rosenbrock (unimodal)
│   ├── Rastrigin, Ackley, Griewank (multimodal)
│   └── Beale, Booth, Matyas (2D visualization)
│
├── visualization_swarm.py           # Visualization tools
│   └── SwarmVisualizer class
│       ├── plot_convergence_curves()
│       ├── plot_performance_comparison()
│       ├── plot_parameter_sensitivity()
│       ├── plot_3d_surface_with_trajectory()
│       ├── plot_statistical_analysis()
│       └── plot_scalability_analysis()
│
├── run_swarm_benchmark.py          # Main benchmark script
│   └── SwarmBenchmark class
│       ├── benchmark_continuous_problems()
│       ├── benchmark_parameter_sensitivity()
│       └── benchmark_scalability()
│
└── README.md                        # This file
```

---

## 🚀 Quick Start

### 1. Generate Test Cases

```bash
# Generate discrete problem instances
python test_cases_discrete.py
# Output: discrete_test_cases.pkl

# Generate continuous benchmark functions
python test_cases_continuous.py
# Output: continuous_test_cases.pkl
```

### 2. Run Comprehensive Benchmark

```bash
python run_swarm_benchmark.py
```

**Output:**
- `swarm_benchmark_results/all_results.pkl` - Raw results
- `swarm_benchmark_results/visualizations/` - All plots
- Console output with summary tables

**Estimated runtime:** 15-30 minutes (30 runs × 5 algorithms × 6 functions)

### 3. Quick Test (Custom)

```python
from test_cases_continuous import SphereFunction
from algorithms.nature_inspired.swarm_based import ParticleSwarmOptimization
from visualization_swarm import SwarmVisualizer

# Create function
func = SphereFunction(dimensions=10)

# Run PSO
pso = ParticleSwarmOptimization(
    population_size=50,
    max_iterations=200,
    bounds=func.bounds,
    seed=42
)
result = pso.optimize(func.get_problem_dict())

print(f"Best fitness: {result.fitness:.6f}")
```

---

## 📊 Visualization Examples

### 1. Convergence Curves

```python
from visualization_swarm import SwarmVisualizer

visualizer = SwarmVisualizer(save_dir='my_results')

# Data: {algorithm_name: [convergence_curves_from_multiple_runs]}
convergence_data = {
    'PSO': [curve1, curve2, curve3, ...],
    'ACO': [curve1, curve2, curve3, ...],
}

visualizer.plot_convergence_curves(
    convergence_data,
    function_name='Sphere-10D',
    save_name='convergence_sphere'
)
```

**Output:**
- Mean convergence with std deviation band
- Best run on log scale
- Saved as PNG and PDF

### 2. Performance Comparison

```python
# Data: {function_name: {algorithm_name: [fitness_values]}}
performance_data = {
    'Sphere-10D': {
        'PSO': [0.001, 0.002, ...],  # 30 runs
        'ACO': [0.003, 0.004, ...],
    },
    'Rastrigin-10D': {
        'PSO': [5.2, 6.1, ...],
        'ACO': [7.3, 8.2, ...],
    }
}

visualizer.plot_performance_comparison(
    performance_data,
    save_name='performance'
)
```

**Output:**
- Box plots per function
- Bar charts with error bars

### 3. Parameter Sensitivity

```python
# Test different parameter values
parameter_values = [0.4, 0.6, 0.7, 0.8, 0.9]  # e.g., inertia weight
results = [
    [fitness_run1, fitness_run2, ...],  # for param=0.4
    [fitness_run1, fitness_run2, ...],  # for param=0.6
    ...
]

visualizer.plot_parameter_sensitivity(
    parameter_name='Inertia Weight',
    parameter_values=parameter_values,
    results=results,
    algorithm_name='PSO',
    function_name='Sphere-10D',
    save_name='sensitivity_inertia'
)
```

**Output:**
- Mean ± std vs parameter value
- Box plots for each value

### 4. 3D Surface with Trajectory (2D functions)

```python
from test_cases_continuous import BealeFunction

func = BealeFunction()  # 2D function
best_positions = [...]  # List of best positions over iterations

visualizer.plot_3d_surface_with_trajectory(
    func=func,
    bounds=(-4.5, 4.5),
    best_positions=best_positions,
    algorithm_name='PSO',
    save_name='trajectory_beale'
)
```

**Output:**
- 3D surface plot with optimization path
- 2D contour plot with path

### 5. Statistical Analysis

```python
visualizer.plot_statistical_analysis(
    performance_data,
    save_name='statistics'
)
```

**Output:**
- Mean, Median, Std, Best, Worst for each algorithm
- Best performer highlighted

### 6. Scalability Analysis

```python
dimensions = [5, 10, 20, 30, 50]
results = {
    'PSO': [0.001, 0.01, 0.1, 1.0, 10.0],
    'ACO': [0.002, 0.02, 0.2, 2.0, 20.0],
}

visualizer.plot_scalability_analysis(
    dimensions=dimensions,
    results=results,
    save_name='scalability'
)
```

**Output:**
- Linear scale plot
- Log-log scale plot

---

## 🔧 Customization

### Custom Benchmark Function

```python
from test_cases_continuous import BenchmarkFunction
import numpy as np

class MyFunction(BenchmarkFunction):
    def __init__(self, dimensions: int = 10):
        bounds = [(-10.0, 10.0)] * dimensions
        super().__init__(
            name=f"MyFunction-{dimensions}D",
            dimensions=dimensions,
            bounds=bounds,
            global_minimum=0.0,
            global_minimum_location=np.zeros(dimensions)
        )
    
    def __call__(self, x: np.ndarray) -> float:
        # Your custom function here
        return np.sum(x**4 - 16*x**2 + 5*x)

# Use it
func = MyFunction(dimensions=10)
problem = func.get_problem_dict()
```

### Custom Algorithm Parameters

```python
from algorithms.nature_inspired.swarm_based import ParticleSwarmOptimization

pso = ParticleSwarmOptimization(
    population_size=100,        # More particles
    max_iterations=500,         # More iterations
    inertia_weight=0.9,         # Higher exploration
    cognitive_coeff=2.0,        # Stronger personal influence
    social_coeff=1.0,           # Weaker swarm influence
    max_velocity=0.5,           # Limit velocity
    verbose=True,               # Show progress
    seed=42                     # Reproducibility
)
```

---

## 📈 Understanding Results

### Convergence Curves
- **Fast initial drop** → Good exploration
- **Smooth plateau** → Converged
- **Oscillations** → Still exploring
- **Compare std bands** → Robustness

### Performance Metrics

```python
# Từ results dictionary:
results['statistics']['Sphere-10D']['PSO']
# {
#     'mean': 0.0012,      # Average over 30 runs
#     'std': 0.0003,       # Standard deviation (lower = more robust)
#     'best': 0.0008,      # Best run (lower bound)
#     'worst': 0.0019,     # Worst run (upper bound)
#     'median': 0.0011     # Middle value
# }
```

### Interpretation
- **Mean**: Overall performance
- **Std**: Robustness (lower is better)
- **Best**: Potential capability
- **Worst**: Worst-case guarantee

### Ranking Algorithms

```python
# Algorithm với lowest mean = best for problem
# Nhưng cũng xem:
# - Std (lower = more reliable)
# - Worst (how bad can it get?)
# - Convergence speed (iterations to reach threshold)
```

---

## 🎯 Recommended Test Scenarios

### 1. Quick Test (5 minutes)
```python
# Test on 2-3 functions, 10 runs each
benchmark = SwarmBenchmark(num_runs=10)
# Only test PSO, ACO
```

### 2. Standard Test (30 minutes)
```python
# Test on 6 functions, 30 runs each
benchmark = SwarmBenchmark(num_runs=30)
# All 5 algorithms
```

### 3. Publication-Quality (2-3 hours)
```python
# Test on 10+ functions, 50 runs each
benchmark = SwarmBenchmark(num_runs=50)
# Include parameter sensitivity, scalability
# Multiple random seeds
```

---

## 📝 Report Integration

### For Your Project Report

#### 1. Algorithm Description Chapter
```markdown
Use visualizations:
- Convergence curves → show convergence behavior
- 3D trajectory → illustrate search process
```

#### 2. Experimental Setup Chapter
```markdown
Cite test cases:
- Table of benchmark functions
- Parameter configurations
- Number of runs, stopping criteria
```

#### 3. Results Chapter
```markdown
Use:
- Performance comparison bar charts
- Statistical analysis tables
- Box plots showing distribution
```

#### 4. Discussion Chapter
```markdown
Use:
- Parameter sensitivity plots → explain tuning
- Scalability plots → analyze complexity
- Convergence comparison → strengths/weaknesses
```

---

## 🐛 Troubleshooting

### Import Errors
```bash
# Make sure you're in the right directory
cd d:\Code\CSTTNT\Team_Project_01\CSTTNT

# Check if files exist
ls test_cases_*.py
ls visualization_swarm.py
```

### Slow Performance
```python
# Reduce num_runs
benchmark = SwarmBenchmark(num_runs=10)  # Instead of 30

# Reduce max_iterations
algorithm_params['max_iterations'] = 100  # Instead of 200

# Test on fewer functions
test_functions = all_functions['unimodal'][:1]  # Just one function
```

### Memory Issues
```python
# For large-scale problems, save results periodically
import pickle

# After each function
with open(f'results_{func_name}.pkl', 'wb') as f:
    pickle.dump(results, f)
```

---

## 📚 References

### Benchmark Functions
1. Sphere: Classic unimodal test
2. Rastrigin: De Jong's F6 function
3. Ackley: Ackley (1987)
4. More: [Virtual Library of Simulation Experiments](https://www.sfu.ca/~ssurjano/optimization.html)

### Algorithms
1. PSO: Kennedy & Eberhart (1995)
2. ACO: Dorigo et al. (2006)
3. ABC: Karaboga & Basturk (2007)
4. FA: Yang (2008)
5. CS: Yang & Deb (2009)

---

## ✅ Checklist for Report

- [ ] Run comprehensive benchmark (30 runs)
- [ ] Generate all visualizations
- [ ] Create convergence plots for each function
- [ ] Create performance comparison charts
- [ ] Run parameter sensitivity analysis
- [ ] Test scalability (different dimensions)
- [ ] Calculate statistical significance (t-test)
- [ ] Document all parameters used
- [ ] Save all results and figures
- [ ] Write analysis and discussion

---

## 💡 Tips for Best Results

1. **Always use same seed** for fair comparison
2. **Run multiple times** (30+ runs) for statistics
3. **Test on diverse problems** (unimodal + multimodal)
4. **Tune parameters** using sensitivity analysis
5. **Save intermediate results** to avoid re-running
6. **Use high-quality figures** (300 dpi) for report
7. **Document everything** in metadata

---

## 🎓 For Your Project

### Minimum Requirements (theo đồ án)
✅ Implement: PSO, ACO, ABC, FA, CS
✅ Test on: Continuous (Sphere, Rastrigin) + Discrete (TSP, Knapsack)
✅ Compare with: BFS, DFS, A*, Hill Climbing, SA
✅ Visualize: Convergence, Performance, Parameter sensitivity
✅ Report: Min 25 pages, analysis, conclusions

### This Package Provides
✅ All 5 swarm algorithms (already implemented)
✅ 10+ continuous test functions
✅ 3 discrete problem types (TSP, Knapsack, Graph Coloring)
✅ Complete visualization suite
✅ Automated benchmarking
✅ Statistical analysis tools
✅ Example usage and integration

---

## 📞 Support

Issues? Check:
1. README examples above
2. Example files in `examples/` directory
3. Comments in source code
4. Project requirements in PDF

---

**Good luck with your project! 🚀**