"""
Example 2: Nature-Inspired Optimization Algorithms
Ví dụ chạy các thuật toán từ Evolution-Based
Genetic Algorithm (GA), Differential Evolution (DE)
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.nature_inspired.evolution_based import GeneticAlgorithm, DifferentialEvolution


def sphere_function(x):
    """
    Sphere Function: f(x) = sum(x_i^2)
    Global minimum at x = [0, 0, ..., 0] with f(x) = 0
    """
    return np.sum(x ** 2)


def rastrigin_function(x):
    """
    Rastrigin Function: Multi-modal, highly complex
    f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
    Global minimum at x = [0, 0, ..., 0] with f(x) = 0
    """
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))


def rosenbrock_function(x):
    """
    Rosenbrock Function: Valley-shaped function
    f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
    Global minimum at x = [1, 1, ..., 1] with f(x) = 0
    """
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def ackley_function(x):
    """
    Ackley Function: Multiple local minima
    Global minimum at x = [0, 0, ..., 0] with f(x) = 0
    """
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2*np.pi*x))
    return (-20 * np.exp(-0.2 * np.sqrt(sum1/n)) 
            - np.exp(sum2/n) + 20 + np.e)


def main():
    print("=" * 70)
    print("NATURE-INSPIRED OPTIMIZATION - Evolution-Based Algorithms")
    print("=" * 70)
    
    # ============= Test Configuration =============
    dimensions = 5  # Number of variables to optimize
    bounds = [(-5.0, 5.0)] * dimensions  # Search space bounds
    
    functions = {
        "Sphere": sphere_function,
        "Rastrigin": rastrigin_function,
        "Rosenbrock": rosenbrock_function,
        "Ackley": ackley_function,
    }
    
    # ============= Genetic Algorithm (GA) =============
    print("\n" + "=" * 70)
    print("GENETIC ALGORITHM (GA)")
    print("=" * 70)
    print(f"Dimensions: {dimensions}")
    print(f"Population Size: 50")
    print(f"Max Generations: 200")
    print(f"Bounds: {bounds[0]}")
    
    ga = GeneticAlgorithm(
        population_size=50,
        max_iterations=200,
        bounds=bounds,
        mutation_rate=0.1,
        crossover_rate=0.8,
        selection_method='tournament',
        crossover_method='uniform',
        verbose=False,
        seed=42
    )
    
    print("\n" + "-" * 70)
    for func_name, func in functions.items():
        print(f"\n{func_name} Function")
        problem = {
            'objective_func': func,
            'dimensions': dimensions,
            'lower_bound': bounds[0][0],
            'upper_bound': bounds[0][1],
            'is_discrete': False
        }
        result_ga = ga.optimize(problem)
        print(f"  Best Fitness: {result_ga.fitness:.6f}")
        print(f"  Best Solution: {result_ga.solution}")
        print(f"  Iterations: {result_ga.iterations}")
        print(f"  Evaluations: {result_ga.evaluations}")
        print(f"  Execution Time: {result_ga.execution_time:.4f}s")
    
    # ============= Differential Evolution (DE) =============
    print("\n" + "=" * 70)
    print("DIFFERENTIAL EVOLUTION (DE)")
    print("=" * 70)
    print(f"Dimensions: {dimensions}")
    print(f"Population Size: 50")
    print(f"Max Generations: 200")
    print(f"Bounds: {bounds[0]}")
    
    de = DifferentialEvolution(
        population_size=50,
        max_iterations=200,
        bounds=bounds,
        mutation_factor=0.8,
        crossover_rate=0.7,
        strategy='rand/1/bin',
        verbose=False,
        seed=42
    )
    
    print("\n" + "-" * 70)
    for func_name, func in functions.items():
        print(f"\n{func_name} Function")
        problem = {
            'objective_func': func,
            'dimensions': dimensions,
            'lower_bound': bounds[0][0],
            'upper_bound': bounds[0][1],
            'is_discrete': False
        }
        result_de = de.optimize(problem)
        print(f"  Best Fitness: {result_de.fitness:.6f}")
        print(f"  Best Solution: {result_de.solution}")
        print(f"  Iterations: {result_de.iterations}")
        print(f"  Evaluations: {result_de.evaluations}")
        print(f"  Execution Time: {result_de.execution_time:.4f}s")
    
    # ============= Detailed Comparison on Sphere Function =============
    print("\n" + "=" * 70)
    print("DETAILED COMPARISON: Sphere Function")
    print("=" * 70)
    
    print("\nOptimizing Sphere Function (Optimal value: 0.0):\n")
    
    # GA
    ga_sphere = GeneticAlgorithm(
        population_size=50,
        max_iterations=300,
        bounds=bounds,
        mutation_rate=0.05,
        crossover_rate=0.85,
        seed=42
    )
    sphere_problem = {
        'objective_func': sphere_function,
        'dimensions': dimensions,
        'lower_bound': bounds[0][0],
        'upper_bound': bounds[0][1],
        'is_discrete': False
    }
    result_ga_sphere = ga_sphere.optimize(sphere_problem)
    
    # DE
    de_sphere = DifferentialEvolution(
        population_size=50,
        max_iterations=300,
        bounds=bounds,
        mutation_factor=0.8,
        crossover_rate=0.7,
        seed=42
    )
    result_de_sphere = de_sphere.optimize(sphere_problem)
    
    print("{:<20} {:<20} {:<20} {:<15}".format(
        "Algorithm", "Best Fitness", "Iterations", "Time (s)"
    ))
    print("-" * 70)
    print("{:<20} {:<20.8f} {:<20} {:<15.4f}".format(
        "Genetic Algorithm", result_ga_sphere.fitness, 
        result_ga_sphere.iterations, result_ga_sphere.execution_time
    ))
    print("{:<20} {:<20.8f} {:<20} {:<15.4f}".format(
        "Differential Evolution", result_de_sphere.fitness,
        result_de_sphere.iterations, result_de_sphere.execution_time
    ))
    
    # Show convergence improvement
    print("\nConvergence Analysis:")
    print(f"GA - Final Fitness: {result_ga_sphere.fitness:.8f}")
    print(f"GA - Convergence Curve Length: {len(result_ga_sphere.convergence_curve)}")
    
    print(f"\nDE - Final Fitness: {result_de_sphere.fitness:.8f}")
    print(f"DE - Convergence Curve Length: {len(result_de_sphere.convergence_curve)}")


if __name__ == "__main__":
    main()
