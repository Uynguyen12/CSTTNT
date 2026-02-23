"""
Test Cases for Continuous Optimization Problems
Benchmark functions for testing swarm algorithms

Bao gồm:
1. Unimodal functions (Sphere, Rosenbrock)
2. Multimodal functions (Rastrigin, Ackley, Griewank)
3. Fixed-dimension functions (Beale, Booth, etc.)
"""

import numpy as np
from typing import Callable, List, Tuple, Dict
import pickle


class BenchmarkFunction:
    """
    Base class cho benchmark functions.
    
    Attributes:
        name: Tên function
        dimensions: Số chiều
        bounds: Search space bounds
        global_minimum: Giá trị optimal
        global_minimum_location: Vị trí optimal
    """
    
    def __init__(self, 
                 name: str,
                 dimensions: int,
                 bounds: List[Tuple[float, float]],
                 global_minimum: float,
                 global_minimum_location: np.ndarray = None):
        self.name = name
        self.dimensions = dimensions
        self.bounds = bounds
        self.global_minimum = global_minimum
        self.global_minimum_location = global_minimum_location
    
    def __call__(self, x: np.ndarray) -> float:
        """Evaluate function."""
        raise NotImplementedError
    
    def get_problem_dict(self) -> dict:
        """Get problem dictionary for optimizers."""
        return {
            'objective_func': self,
            'dimensions': self.dimensions,
            'lower_bound': self.bounds[0][0],
            'upper_bound': self.bounds[0][1],
            'is_discrete': False
        }


# ============= Unimodal Functions =============

class SphereFunction(BenchmarkFunction):
    """
    Sphere Function: f(x) = sum(x_i^2)
    
    Characteristics:
    - Unimodal
    - Convex
    - Separable
    - Global minimum: f(0, ..., 0) = 0
    
    Best for testing: Convergence speed
    """
    
    def __init__(self, dimensions: int = 10):
        bounds = [(-5.12, 5.12)] * dimensions
        super().__init__(
            name=f"Sphere-{dimensions}D",
            dimensions=dimensions,
            bounds=bounds,
            global_minimum=0.0,
            global_minimum_location=np.zeros(dimensions)
        )
    
    def __call__(self, x: np.ndarray) -> float:
        return np.sum(x ** 2)


class RosenbrockFunction(BenchmarkFunction):
    """
    Rosenbrock Function (Valley-shaped)
    f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
    
    Characteristics:
    - Unimodal
    - Non-convex (narrow valley)
    - Non-separable
    - Global minimum: f(1, ..., 1) = 0
    
    Best for testing: Exploitation in narrow valleys
    """
    
    def __init__(self, dimensions: int = 10):
        bounds = [(-5.0, 10.0)] * dimensions
        super().__init__(
            name=f"Rosenbrock-{dimensions}D",
            dimensions=dimensions,
            bounds=bounds,
            global_minimum=0.0,
            global_minimum_location=np.ones(dimensions)
        )
    
    def __call__(self, x: np.ndarray) -> float:
        return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


# ============= Multimodal Functions =============

class RastriginFunction(BenchmarkFunction):
    """
    Rastrigin Function (Highly multimodal)
    f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
    
    Characteristics:
    - Highly multimodal
    - Separable
    - Regular structure of local minima
    - Global minimum: f(0, ..., 0) = 0
    
    Best for testing: Ability to escape local minima
    """
    
    def __init__(self, dimensions: int = 10):
        bounds = [(-5.12, 5.12)] * dimensions
        super().__init__(
            name=f"Rastrigin-{dimensions}D",
            dimensions=dimensions,
            bounds=bounds,
            global_minimum=0.0,
            global_minimum_location=np.zeros(dimensions)
        )
    
    def __call__(self, x: np.ndarray) -> float:
        A = 10
        n = len(x)
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))


class AckleyFunction(BenchmarkFunction):
    """
    Ackley Function (Many local minima)
    
    Characteristics:
    - Highly multimodal
    - Nearly flat outer region
    - Large number of local minima
    - Global minimum: f(0, ..., 0) = 0
    
    Best for testing: Global exploration capability
    """
    
    def __init__(self, dimensions: int = 10):
        bounds = [(-32.768, 32.768)] * dimensions
        super().__init__(
            name=f"Ackley-{dimensions}D",
            dimensions=dimensions,
            bounds=bounds,
            global_minimum=0.0,
            global_minimum_location=np.zeros(dimensions)
        )
    
    def __call__(self, x: np.ndarray) -> float:
        n = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(2*np.pi*x))
        return (-20 * np.exp(-0.2 * np.sqrt(sum1/n)) 
                - np.exp(sum2/n) + 20 + np.e)


class GriewankFunction(BenchmarkFunction):
    """
    Griewank Function (Regularly distributed minima)
    f(x) = 1 + sum(x_i^2/4000) - prod(cos(x_i/sqrt(i)))
    
    Characteristics:
    - Multimodal
    - Product term creates correlations between dimensions
    - Numerous local minima
    - Global minimum: f(0, ..., 0) = 0
    
    Best for testing: Handling correlation between dimensions
    """
    
    def __init__(self, dimensions: int = 10):
        bounds = [(-600.0, 600.0)] * dimensions
        super().__init__(
            name=f"Griewank-{dimensions}D",
            dimensions=dimensions,
            bounds=bounds,
            global_minimum=0.0,
            global_minimum_location=np.zeros(dimensions)
        )
    
    def __call__(self, x: np.ndarray) -> float:
        sum_term = np.sum(x**2 / 4000)
        prod_term = 1
        for i, xi in enumerate(x):
            prod_term *= np.cos(xi / np.sqrt(i + 1))
        return 1 + sum_term - prod_term


class SchwefelFunction(BenchmarkFunction):
    """
    Schwefel Function (Highly deceptive)
    f(x) = 418.9829*n - sum(x_i * sin(sqrt(|x_i|)))
    
    Characteristics:
    - Highly multimodal
    - Deceptive (global optimum far from local optima)
    - Global minimum: f(420.9687, ..., 420.9687) ≈ 0
    
    Best for testing: Resistance to premature convergence
    """
    
    def __init__(self, dimensions: int = 10):
        bounds = [(-500.0, 500.0)] * dimensions
        global_min_loc = np.full(dimensions, 420.9687)
        super().__init__(
            name=f"Schwefel-{dimensions}D",
            dimensions=dimensions,
            bounds=bounds,
            global_minimum=0.0,
            global_minimum_location=global_min_loc
        )
    
    def __call__(self, x: np.ndarray) -> float:
        n = len(x)
        return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))


class LevyFunction(BenchmarkFunction):
    """
    Levy Function (Many local minima)
    
    Characteristics:
    - Multimodal
    - Many local minima
    - Global minimum: f(1, ..., 1) = 0
    
    Best for testing: Fine-tuning capability
    """
    
    def __init__(self, dimensions: int = 10):
        bounds = [(-10.0, 10.0)] * dimensions
        super().__init__(
            name=f"Levy-{dimensions}D",
            dimensions=dimensions,
            bounds=bounds,
            global_minimum=0.0,
            global_minimum_location=np.ones(dimensions)
        )
    
    def __call__(self, x: np.ndarray) -> float:
        w = 1 + (x - 1) / 4
        
        term1 = np.sin(np.pi * w[0])**2
        term2 = np.sum((w[:-1] - 1)**2 * (1 + 10*np.sin(np.pi*w[:-1] + 1)**2))
        term3 = (w[-1] - 1)**2 * (1 + np.sin(2*np.pi*w[-1])**2)
        
        return term1 + term2 + term3


# ============= Fixed-Dimension Functions (for 2D visualization) =============

class BealeFunction(BenchmarkFunction):
    """
    Beale Function (2D)
    
    Characteristics:
    - Multimodal
    - Plate-shaped
    - Global minimum: f(3, 0.5) = 0
    """
    
    def __init__(self):
        bounds = [(-4.5, 4.5), (-4.5, 4.5)]
        super().__init__(
            name="Beale-2D",
            dimensions=2,
            bounds=bounds,
            global_minimum=0.0,
            global_minimum_location=np.array([3.0, 0.5])
        )
    
    def __call__(self, x: np.ndarray) -> float:
        x1, x2 = x[0], x[1]
        term1 = (1.5 - x1 + x1*x2)**2
        term2 = (2.25 - x1 + x1*x2**2)**2
        term3 = (2.625 - x1 + x1*x2**3)**2
        return term1 + term2 + term3


class BoothFunction(BenchmarkFunction):
    """
    Booth Function (2D)
    
    Characteristics:
    - Unimodal
    - Global minimum: f(1, 3) = 0
    """
    
    def __init__(self):
        bounds = [(-10.0, 10.0), (-10.0, 10.0)]
        super().__init__(
            name="Booth-2D",
            dimensions=2,
            bounds=bounds,
            global_minimum=0.0,
            global_minimum_location=np.array([1.0, 3.0])
        )
    
    def __call__(self, x: np.ndarray) -> float:
        x1, x2 = x[0], x[1]
        return (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2


class MatyasFunction(BenchmarkFunction):
    """
    Matyas Function (2D)
    
    Characteristics:
    - Unimodal
    - Plate-shaped with ripples
    - Global minimum: f(0, 0) = 0
    """
    
    def __init__(self):
        bounds = [(-10.0, 10.0), (-10.0, 10.0)]
        super().__init__(
            name="Matyas-2D",
            dimensions=2,
            bounds=bounds,
            global_minimum=0.0,
            global_minimum_location=np.zeros(2)
        )
    
    def __call__(self, x: np.ndarray) -> float:
        x1, x2 = x[0], x[1]
        return 0.26*(x1**2 + x2**2) - 0.48*x1*x2


# ============= Test Case Collections =============

def get_all_benchmark_functions() -> Dict[str, List[BenchmarkFunction]]:
    """
    Get all benchmark functions organized by category.
    
    Returns:
        Dictionary with categories as keys
    """
    return {
        'unimodal': [
            SphereFunction(dimensions=5),
            SphereFunction(dimensions=10),
            SphereFunction(dimensions=20),
            RosenbrockFunction(dimensions=5),
            RosenbrockFunction(dimensions=10),
        ],
        'multimodal': [
            RastriginFunction(dimensions=5),
            RastriginFunction(dimensions=10),
            RastriginFunction(dimensions=20),
            AckleyFunction(dimensions=5),
            AckleyFunction(dimensions=10),
            GriewankFunction(dimensions=10),
            SchwefelFunction(dimensions=5),
            SchwefelFunction(dimensions=10),
            LevyFunction(dimensions=10),
        ],
        'fixed_2d': [
            BealeFunction(),
            BoothFunction(),
            MatyasFunction(),
        ]
    }


def generate_continuous_test_cases():
    """Generate all continuous test cases."""
    print("=" * 70)
    print("GENERATING CONTINUOUS OPTIMIZATION TEST CASES")
    print("=" * 70)
    
    all_functions = get_all_benchmark_functions()
    
    for category, functions in all_functions.items():
        print(f"\n{category.upper()} FUNCTIONS:")
        print("-" * 70)
        for func in functions:
            print(f"  {func.name:30s} | "
                  f"Bounds: {func.bounds[0]} | "
                  f"Global Min: {func.global_minimum}")
    
    # Save test cases
    print("\n" + "=" * 70)
    print("Saving test cases...")
    with open('continuous_test_cases.pkl', 'wb') as f:
        pickle.dump(all_functions, f)
    
    print("Test cases saved to 'continuous_test_cases.pkl'")
    print("=" * 70)
    
    return all_functions


def main():
    """Generate and test continuous benchmark functions."""
    test_cases = generate_continuous_test_cases()
    
    # Test each function at global minimum
    print("\n" + "=" * 70)
    print("TESTING FUNCTIONS AT GLOBAL MINIMUM")
    print("=" * 70)
    
    for category, functions in test_cases.items():
        print(f"\n{category.upper()}:")
        for func in functions:
            if func.global_minimum_location is not None:
                value = func(func.global_minimum_location)
                error = abs(value - func.global_minimum)
                print(f"  {func.name:30s} | "
                      f"Expected: {func.global_minimum:.6f} | "
                      f"Got: {value:.6f} | "
                      f"Error: {error:.2e}")
    
    return test_cases


if __name__ == "__main__":
    test_cases = main()