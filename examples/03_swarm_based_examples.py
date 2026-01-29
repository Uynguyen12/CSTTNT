"""
Example 3: Swarm-Based Nature-Inspired Algorithms
Ví dụ chạy các thuật toán từ Swarm-Based
Particle Swarm Optimization (PSO), Ant Colony Optimization (ACO), etc.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.nature_inspired.swarm_based import (
    ParticleSwarmOptimization,
    AntColonyOptimization
)


# ============= Objective Functions =============
def sphere_function(x):
    """Sphere Function: f(x) = sum(x_i^2), minimum at [0, 0, ..., 0]"""
    return np.sum(x ** 2)


def beale_function(x):
    """
    Beale Function (2D): Multimodal
    f(x,y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2
    """
    x_val, y_val = x[0], x[1]
    term1 = (1.5 - x_val + x_val*y_val)**2
    term2 = (2.25 - x_val + x_val*y_val**2)**2
    term3 = (2.625 - x_val + x_val*y_val**3)**2
    return term1 + term2 + term3


def booth_function(x):
    """
    Booth Function (2D): Multiple local minima
    f(x,y) = (x + 2y - 7)^2 + (2x + y - 5)^2
    Global minimum at (1, 3) with f = 0
    """
    x_val, y_val = x[0], x[1]
    return (x_val + 2*y_val - 7)**2 + (2*x_val + y_val - 5)**2


def matyas_function(x):
    """
    Matyas Function (2D): Plate-shaped with ripples
    f(x,y) = 0.26(x^2 + y^2) - 0.48*x*y
    """
    x_val, y_val = x[0], x[1]
    return 0.26*(x_val**2 + y_val**2) - 0.48*x_val*y_val


def main():
    print("=" * 70)
    print("SWARM-BASED NATURE-INSPIRED ALGORITHMS")
    print("=" * 70)
    
    # ============= Particle Swarm Optimization (PSO) =============
    print("\n" + "=" * 70)
    print("PARTICLE SWARM OPTIMIZATION (PSO)")
    print("=" * 70)
    
    # PSO for continuous optimization (high-dimensional)
    print("\n--- High-Dimensional Sphere Function (5D) ---")
    bounds_5d = [(-5.0, 5.0)] * 5
    
    pso_sphere = ParticleSwarmOptimization(
        population_size=30,
        max_iterations=200,
        bounds=bounds_5d,
        inertia_weight=0.7,
        cognitive_coeff=1.5,
        social_coeff=1.5,
        max_velocity=0.2,
        verbose=False,
        seed=42
    )
    
    sphere_problem_5d = {
        'objective_func': sphere_function,
        'dimensions': 5,
        'lower_bound': -5.0,
        'upper_bound': 5.0,
        'is_discrete': False
    }
    result_pso_sphere = pso_sphere.optimize(sphere_problem_5d)
    print(f"Best Fitness: {result_pso_sphere.fitness:.8f}")
    print(f"Iterations: {result_pso_sphere.iterations}")
    print(f"Execution Time: {result_pso_sphere.execution_time:.4f}s")
    print(f"Best Solution: {result_pso_sphere.solution}")
    
    # PSO for 2D functions
    print("\n--- 2D Test Functions ---")
    
    functions_2d = {
        "Beale": (beale_function, [(-4.5, 4.5), (-4.5, 4.5)]),
        "Booth": (booth_function, [(-10, 10), (-10, 10)]),
        "Matyas": (matyas_function, [(-10, 10), (-10, 10)]),
    }
    
    for func_name, (func, bounds) in functions_2d.items():
        print(f"\n{func_name} Function:")
        pso = ParticleSwarmOptimization(
            population_size=25,
            max_iterations=200,
            bounds=bounds,
            inertia_weight=0.7,
            cognitive_coeff=1.5,
            social_coeff=1.5,
            seed=42
        )
        
        func_problem = {
            'objective_func': func,
            'dimensions': 2,
            'lower_bound': bounds[0][0],
            'upper_bound': bounds[0][1],
            'is_discrete': False
        }
        result = pso.optimize(func_problem)
        print(f"  Best Fitness: {result.fitness:.6f}")
        print(f"  Best Solution: [{result.solution[0]:.4f}, {result.solution[1]:.4f}]")
        print(f"  Iterations: {result.iterations}")
    
    # ============= Ant Colony Optimization (ACO) =============
    print("\n" + "=" * 70)
    print("ANT COLONY OPTIMIZATION (ACO)")
    print("=" * 70)
    
    print("\nACO for Traveling Salesman Problem (TSP) Style:")
    print("-" * 70)
    
    # ACO for continuous optimization (ACOR - Ant Colony Optimization for continuous domains)
    print("\nACO for Continuous Optimization (5D Sphere):")
    
    aco = AntColonyOptimization(
        population_size=20,
        max_iterations=200,
        archive_size=20,
        intensification_factor=0.85,
        deviation_distance=0.1,
        verbose=False,
        seed=42
    )
    
    aco_problem = {
        'objective_func': sphere_function,
        'dimensions': 5,
        'lower_bound': -5.0,
        'upper_bound': 5.0,
        'is_discrete': False
    }
    result_aco = aco.optimize(aco_problem)
    print(f"Best Fitness: {result_aco.fitness:.8f}")
    print(f"Best Solution: {result_aco.solution}")
    print(f"Iterations: {result_aco.iterations}")
    print(f"Execution Time: {result_aco.execution_time:.4f}s")
    
    # ============= Comparison Summary =============
    print("\n" + "=" * 70)
    print("ALGORITHM COMPARISON")
    print("=" * 70)
    
    print("\nOptimizing Sphere Function (5D) - 200 iterations, 30 particles/ants:")
    
    pso_result = ParticleSwarmOptimization(
        population_size=30,
        max_iterations=200,
        bounds=bounds_5d,
        seed=42
    ).optimize(sphere_problem_5d)
    
    print(f"\nPSO Performance:")
    print(f"  Final Fitness: {pso_result.fitness:.8f}")
    print(f"  Execution Time: {pso_result.execution_time:.4f}s")
    print(f"  Convergence: {len(pso_result.convergence_curve)} iterations")
    
    print(f"\nPSO Parameters Used:")
    print(f"  Inertia Weight: 0.7")
    print(f"  Cognitive Coefficient: 1.5")
    print(f"  Social Coefficient: 1.5")
    print(f"  Number of Particles: 30")


if __name__ == "__main__":
    main()
