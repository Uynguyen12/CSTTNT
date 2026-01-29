"""
Example 4: Physics-Based and Human-Based Nature-Inspired Algorithms
Ví dụ chạy các thuật toán từ Physics-Based và Human-Based
Simulated Annealing, Hill Climbing, Gravitational Search, Harmony Search, etc.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.nature_inspired.physics_based import (
    SimulatedAnnealing,
    HillClimbing,
    HarmonySearch
)


# ============= Objective Functions =============
def griewank_function(x):
    """
    Griewank Function: Highly multimodal
    f(x) = 1 + sum(x_i^2/4000) - prod(cos(x_i/sqrt(i)))
    """
    sum_term = np.sum(x**2 / 4000)
    prod_term = 1
    for i, xi in enumerate(x):
        prod_term *= np.cos(xi / np.sqrt(i + 1))
    return 1 + sum_term - prod_term


def schwefel_function(x):
    """
    Schwefel Function: Highly deceptive
    f(x) = 418.9829*n - sum(x_i * sin(sqrt(|x_i|)))
    """
    n = len(x)
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def levy_function(x):
    """
    Levy Function: Many local minima
    Has a global minimum at [1, 1, ..., 1] with f = 0
    """
    d = len(x)
    w = 1 + (x - 1) / 4
    
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10*np.sin(np.pi*w[:-1] + 1)**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2*np.pi*w[-1])**2)
    
    return term1 + term2 + term3


def simple_2d_function(x):
    """Simple 2D function with multiple minima"""
    x_val, y_val = x[0], x[1]
    return (x_val - 2)**2 + (y_val - 3)**2 + 5*np.sin(x_val*y_val)


def main():
    print("=" * 70)
    print("PHYSICS-BASED & HUMAN-BASED NATURE-INSPIRED ALGORITHMS")
    print("=" * 70)
    
    # ============= Simulated Annealing (SA) =============
    print("\n" + "=" * 70)
    print("SIMULATED ANNEALING (SA)")
    print("=" * 70)
    print("Physics-inspired: Models cooling process of materials")
    
    bounds_3d = [(-5.0, 5.0)] * 3
    
    print("\n--- Griewank Function (3D) ---")
    sa_griewank = SimulatedAnnealing(
        bounds=bounds_3d,
        max_iterations=1000,
        initial_temperature=100.0,
        cooling_rate=0.95,
        temperature_min=0.01,
        perturbation_scale=0.5,
        verbose=False,
        seed=42
    )
    
    result_sa_griewank = sa_griewank.optimize({
        'objective_func': griewank_function,
        'dimensions': 3,
        'lower_bound': -5.0,
        'upper_bound': 5.0,
        'is_discrete': False
    })
    print(f"Best Fitness: {result_sa_griewank.fitness:.8f}")
    print(f"Iterations: {result_sa_griewank.iterations}")
    print(f"Execution Time: {result_sa_griewank.execution_time:.4f}s")
    print(f"Best Solution: {result_sa_griewank.solution}")
    
    # SA for 2D function
    print("\n--- 2D Custom Function ---")
    bounds_2d = [(-10.0, 10.0), (-10.0, 10.0)]
    
    sa_2d = SimulatedAnnealing(
        bounds=bounds_2d,
        max_iterations=500,
        initial_temperature=50.0,
        cooling_rate=0.97,
        perturbation_scale=0.3,
        seed=42
    )
    
    result_sa_2d = sa_2d.optimize({
        'objective_func': simple_2d_function,
        'dimensions': 2,
        'lower_bound': -10.0,
        'upper_bound': 10.0,
        'is_discrete': False
    })
    print(f"Best Fitness: {result_sa_2d.fitness:.6f}")
    print(f"Best Solution: [{result_sa_2d.solution[0]:.4f}, {result_sa_2d.solution[1]:.4f}]")
    print(f"Iterations: {result_sa_2d.iterations}")
    
    # ============= Hill Climbing =============
    print("\n" + "=" * 70)
    print("HILL CLIMBING")
    print("=" * 70)
    print("Greedy local search approach")
    
    print("\n--- Griewank Function (3D) ---")
    hc_griewank = HillClimbing(
        bounds=bounds_3d,
        max_iterations=1000,
        step_size=0.5,
        step_size_reduction=0.99,
        step_size_min=0.001,
        verbose=False,
        seed=42
    )
    
    result_hc_griewank = hc_griewank.optimize({
        'objective_func': griewank_function,
        'dimensions': 3,
        'lower_bound': -5.0,
        'upper_bound': 5.0,
        'is_discrete': False
    })
    print(f"Best Fitness: {result_hc_griewank.fitness:.8f}")
    print(f"Iterations: {result_hc_griewank.iterations}")
    print(f"Execution Time: {result_hc_griewank.execution_time:.4f}s")
    print(f"Best Solution: {result_hc_griewank.solution}")
    
    print("\n--- 2D Custom Function ---")
    hc_2d = HillClimbing(
        bounds=bounds_2d,
        max_iterations=500,
        step_size=0.5,
        seed=42
    )
    
    result_hc_2d = hc_2d.optimize({
        'objective_func': simple_2d_function,
        'dimensions': 2,
        'lower_bound': -10.0,
        'upper_bound': 10.0,
        'is_discrete': False
    })
    print(f"Best Fitness: {result_hc_2d.fitness:.6f}")
    print(f"Best Solution: [{result_hc_2d.solution[0]:.4f}, {result_hc_2d.solution[1]:.4f}]")
    print(f"Iterations: {result_hc_2d.iterations}")
    
    # ============= Harmony Search (Human-Based) =============
    print("\n" + "=" * 70)
    print("HARMONY SEARCH (HS)")
    print("=" * 70)
    print("Human-inspired: Models musicians creating harmony")
    
    bounds_4d = [(-5.0, 5.0)] * 4
    
    print("\n--- Levy Function (4D) ---")
    hs_levy = HarmonySearch(
        harmony_memory_size=20,
        max_iterations=200,
        bounds=bounds_4d,
        hmcr=0.9,
        par=0.3,
        bandwidth=0.5,
        verbose=False,
        seed=42
    )
    
    result_hs_levy = hs_levy.optimize({
        'objective_func': levy_function,
        'dimensions': 4,
        'lower_bound': -5.0,
        'upper_bound': 5.0,
        'is_discrete': False
    })
    print(f"Best Fitness: {result_hs_levy.fitness:.8f}")
    print(f"Iterations: {result_hs_levy.iterations}")
    print(f"Execution Time: {result_hs_levy.execution_time:.4f}s")
    print(f"Best Solution: {result_hs_levy.solution}")
    
    # HS for 2D function
    print("\n--- 2D Custom Function ---")
    hs_2d = HarmonySearch(
        harmony_memory_size=15,
        max_iterations=300,
        bounds=bounds_2d,
        hmcr=0.9,
        par=0.3,
        seed=42
    )
    
    result_hs_2d = hs_2d.optimize({
        'objective_func': simple_2d_function,
        'dimensions': 2,
        'lower_bound': -10.0,
        'upper_bound': 10.0,
        'is_discrete': False
    })
    print(f"Best Fitness: {result_hs_2d.fitness:.6f}")
    print(f"Best Solution: [{result_hs_2d.solution[0]:.4f}, {result_hs_2d.solution[1]:.4f}]")
    print(f"Iterations: {result_hs_2d.iterations}")
    
    # ============= Comprehensive Comparison =============
    print("\n" + "=" * 70)
    print("COMPREHENSIVE ALGORITHM COMPARISON")
    print("=" * 70)
    
    print("\nOptimizing Griewank Function (3D) - 1000 iterations:")
    print("-" * 70)
    
    sa = SimulatedAnnealing(bounds=bounds_3d, max_iterations=1000, seed=42)
    hc = HillClimbing(bounds=bounds_3d, max_iterations=1000, seed=42)
    hs = HarmonySearch(bounds=bounds_3d, max_iterations=1000, seed=42)
    
    result_sa = sa.optimize({
        'objective_func': griewank_function,
        'dimensions': 3,
        'lower_bound': -5.0,
        'upper_bound': 5.0,
        'is_discrete': False
    })
    result_hc = hc.optimize({
        'objective_func': griewank_function,
        'dimensions': 3,
        'lower_bound': -5.0,
        'upper_bound': 5.0,
        'is_discrete': False
    })
    result_hs = hs.optimize({
        'objective_func': griewank_function,
        'dimensions': 3,
        'lower_bound': -5.0,
        'upper_bound': 5.0,
        'is_discrete': False
    })
    
    print("\n{:<25} {:<20} {:<15}".format("Algorithm", "Best Fitness", "Time (s)"))
    print("-" * 70)
    print("{:<25} {:<20.8f} {:<15.4f}".format(
        "Simulated Annealing", result_sa.fitness, result_sa.execution_time
    ))
    print("{:<25} {:<20.8f} {:<15.4f}".format(
        "Hill Climbing", result_hc.fitness, result_hc.execution_time
    ))
    print("{:<25} {:<20.8f} {:<15.4f}".format(
        "Harmony Search", result_hs.fitness, result_hs.execution_time
    ))
    
    # Analysis
    print("\nAnalysis:")
    print(f"Best Algorithm: {min([('SA', result_sa.fitness), ('HC', result_hc.fitness), ('HS', result_hs.fitness)], key=lambda x: x[1])[0]}")
    print("\nCharacteristics:")
    print("- Simulated Annealing: Good for escaping local minima (simulates cooling)")
    print("- Hill Climbing: Fast but often gets stuck in local minima")
    print("- Harmony Search: Balanced exploration-exploitation from music theory")


if __name__ == "__main__":
    main()
