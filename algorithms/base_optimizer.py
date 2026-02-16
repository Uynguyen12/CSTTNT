"""
Base Optimizer Class
Lớp cha trừu tượng cho tất cả các thuật toán tối ưu hóa
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import time


# ==========================
# Optimization Results
# ==========================

@dataclass
class OptimizationResults:
    solution: Any = None
    fitness: float = float('inf')
    success: bool = False
    iterations: int = 0
    evaluations: int = 0
    execution_time: float = 0.0
    convergence_curve: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"OptimizationResults("
                f"success={self.success}, "
                f"fitness={self.fitness:.6f}, "
                f"iterations={self.iterations}, "
                f"time={self.execution_time:.4f}s)")


# ==========================
# Base Optimizer
# ==========================

class BaseOptimizer(ABC):

    def __init__(self,
                 max_iterations: int = 1000,
                 tolerance: float = 1e-6,
                 verbose: bool = False,
                 seed: Optional[int] = None,
                 is_maximization: bool = False,
                 enable_animation: bool = True):

        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        self.seed = seed
        self.is_maximization = is_maximization
        self.enable_animation = enable_animation

        # Internal state
        self._iteration = 0
        self._evaluations = 0
        self._best_fitness = float('-inf') if is_maximization else float('inf')
        self._convergence_history = []
        self._start_time = 0.0

        # Animation histories
        self.position_history = []
        self.best_route_history = []
        self.fitness_history = []

        if seed is not None:
            np.random.seed(seed)

    # ==========================
    # Abstract methods
    # ==========================

    @abstractmethod
    def _initialize(self, problem: Any) -> None:
        pass

    @abstractmethod
    def _iterate(self) -> float:
        pass

    @abstractmethod
    def _get_best_solution(self) -> Tuple[Any, float]:
        pass

    # ==========================
    # Utility methods
    # ==========================

    def _is_better(self, new: float, old: float) -> bool:
        return new > old if self.is_maximization else new < old

    def _reset_state(self):
        self._iteration = 0
        self._evaluations = 0
        self._best_fitness = float('-inf') if self.is_maximization else float('inf')
        self._convergence_history.clear()

        self.position_history.clear()
        self.best_route_history.clear()
        self.fitness_history.clear()

        if self.seed is not None:
            np.random.seed(self.seed)

    def _record_state(self,
                      population=None,
                      best_solution=None,
                      best_fitness=None):

        if not self.enable_animation:
            return

        # Population-based history
        if population is not None:
            self.position_history.append(np.copy(population))

        # Best solution history
        if best_solution is not None:
            if isinstance(best_solution, np.ndarray):
                self.best_route_history.append(np.copy(best_solution))
            elif hasattr(best_solution, "copy"):
                self.best_route_history.append(best_solution.copy())
            else:
                self.best_route_history.append(best_solution)

        # Fitness history
        if best_fitness is not None:
            self.fitness_history.append(best_fitness)

    # ==========================
    # Main optimize loop
    # ==========================

    def optimize(self, problem: Any) -> OptimizationResults:

        self._reset_state()
        self._start_time = time.perf_counter()

        # Initialize algorithm
        self._initialize(problem)

        # Record initial state
        best_solution, best_fitness = self._get_best_solution()

        self._record_state(
            population=getattr(self, "population", None),
            best_solution=best_solution,
            best_fitness=best_fitness
        )

        prev_fitness = best_fitness
        stagnant_iterations = 0
        max_stagnant = 50

        while self._iteration < self.max_iterations:

            current_best = self._iterate()
            self._iteration += 1

            self._convergence_history.append(current_best)

            improvement = abs(prev_fitness - current_best)

            if improvement < self.tolerance:
                stagnant_iterations += 1
            else:
                stagnant_iterations = 0

            # Record state after iteration
            best_solution, best_fitness = self._get_best_solution()

            self._record_state(
                population=getattr(self, "population", None),
                best_solution=best_solution,
                best_fitness=best_fitness
            )

            if self.verbose and self._iteration % 10 == 0:
                self._print_progress(current_best)

            if stagnant_iterations >= max_stagnant:
                if self.verbose:
                    print(f"\n[Early Stop] No improvement for {max_stagnant} iterations")
                break

            prev_fitness = current_best

        execution_time = time.perf_counter() - self._start_time

        results = OptimizationResults(
            solution=best_solution,
            fitness=best_fitness,
            success=True,
            iterations=self._iteration,
            evaluations=self._evaluations,
            execution_time=execution_time,
            convergence_curve=self._convergence_history,
            metadata=self._get_metadata()
        )

        if self.verbose:
            self._print_final_results(results)

        return results

    # ==========================
    # Logging
    # ==========================

    def _print_progress(self, current_best: float):
        print(f"Iteration {self._iteration:4d} | "
              f"Best Fitness: {current_best:.6f} | "
              f"Evaluations: {self._evaluations}")

    def _print_final_results(self, results: OptimizationResults):
        print("\n" + "="*70)
        print(f"Optimization Complete: {self.__class__.__name__}")
        print("="*70)
        print(f"Success: {results.success}")
        print(f"Best Fitness: {results.fitness:.6f}")
        print(f"Iterations: {results.iterations}")
        print(f"Function Evaluations: {results.evaluations}")
        print(f"Execution Time: {results.execution_time:.4f} seconds")
        print("="*70)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            'algorithm': self.__class__.__name__,
            'max_iterations': self.max_iterations,
            'tolerance': self.tolerance,
            'seed': self.seed,
            'is_maximization': self.is_maximization
        }


# ==========================
# Population-Based Optimizer
# ==========================

class PopulationBasedOptimizer(BaseOptimizer):

    def __init__(self,
                 population_size: int = 50,
                 max_iterations: int = 1000,
                 tolerance: float = 1e-6,
                 verbose: bool = False,
                 seed: Optional[int] = None,
                 is_maximization: bool = False,
                 enable_animation: bool = True):

        super().__init__(max_iterations, tolerance, verbose,
                         seed, is_maximization, enable_animation)

        self.population_size = population_size
        self.population = None

    def _initialize_random_population(self,
                                      dimensions: int,
                                      lower_bound: float,
                                      upper_bound: float) -> np.ndarray:

        return np.random.uniform(
            lower_bound,
            upper_bound,
            size=(self.population_size, dimensions)
        )

    def _evaluate_population(self,
                             population: np.ndarray,
                             objective_func) -> np.ndarray:

        fitnesses = np.array([objective_func(ind) for ind in population])
        self._evaluations += len(population)
        return fitnesses

    def _get_metadata(self) -> Dict[str, Any]:
        metadata = super()._get_metadata()
        metadata['population_size'] = self.population_size
        return metadata


# ==========================
# Single-Solution Optimizer
# ==========================

class SingleSolutionOptimizer(BaseOptimizer):

    def __init__(self,
                 max_iterations: int = 1000,
                 tolerance: float = 1e-6,
                 verbose: bool = False,
                 seed: Optional[int] = None,
                 is_maximization: bool = False,
                 enable_animation: bool = True):

        super().__init__(max_iterations, tolerance, verbose,
                         seed, is_maximization, enable_animation)

        self.current_solution = None
        self.current_fitness = float('-inf') if is_maximization else float('inf')

    def _initialize_random_solution(self,
                                    dimensions: int,
                                    lower_bound: float,
                                    upper_bound: float) -> np.ndarray:

        return np.random.uniform(lower_bound, upper_bound, size=dimensions)

    def _evaluate_solution(self, solution: np.ndarray, objective_func) -> float:
        self._evaluations += 1
        return objective_func(solution)
