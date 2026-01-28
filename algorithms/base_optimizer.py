"""
Base Optimizer Class
Lớp cha trừu tượng cho tất cả các thuật toán tối ưu hóa
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import time


@dataclass
class OptimizationResults:
    """
    Container cho kết quả của thuật toán tối ưu hóa.
    
    Attributes:
        solution: Best solution found (có thể là path, vector, hoặc combination)
        fitness: Fitness/cost của solution
        success: Có tìm thấy solution hợp lệ không
        iterations: Số iterations đã chạy
        evaluations: Số lần evaluate objective function
        execution_time: Thời gian chạy (seconds)
        convergence_curve: History của best fitness theo iterations
        metadata: Dictionary chứa thông tin bổ sung
    """
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
                f"fitness={self.fitness:.4f}, "
                f"iterations={self.iterations}, "
                f"time={self.execution_time:.4f}s)")


class BaseOptimizer(ABC):
    """
    Lớp cha trừu tượng cho tất cả algorithms.
    
    Các subclass phải implement:
        - _initialize(): Khởi tạo population/state
        - _iterate(): Thực hiện một iteration
        - _get_best_solution(): Trả về solution tốt nhất hiện tại
    """
    
    def __init__(self, 
                 max_iterations: int = 1000,
                 tolerance: float = 1e-6,
                 verbose: bool = False,
                 seed: Optional[int] = None):
        """
        Args:
            max_iterations: Số iterations tối đa
            tolerance: Ngưỡng để coi như đã converge
            verbose: In thông tin debug không
            seed: Random seed để reproduce kết quả
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        self.seed = seed
        
        # Internal state
        self._iteration = 0
        self._evaluations = 0
        self._best_fitness = float('inf')
        self._convergence_history = []
        self._start_time = 0.0
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
    
    @abstractmethod
    def _initialize(self, problem: Any) -> None:
        """
        Khởi tạo algorithm state (population, parameters, etc.).
        
        Args:
            problem: Problem object để optimize
        """
        pass
    
    @abstractmethod
    def _iterate(self) -> float:
        """
        Thực hiện một iteration của algorithm.
        
        Returns:
            best_fitness: Best fitness found trong iteration này
        """
        pass
    
    @abstractmethod
    def _get_best_solution(self) -> Tuple[Any, float]:
        """
        Trả về solution tốt nhất hiện tại.
        
        Returns:
            (solution, fitness): Best solution và fitness của nó
        """
        pass
    
    def optimize(self, problem: Any) -> OptimizationResults:
        """
        Chạy thuật toán tối ưu hóa.
        
        Args:
            problem: Problem object để optimize
            
        Returns:
            OptimizationResults: Kết quả tối ưu hóa
        """
        self._start_time = time.perf_counter()
        
        # Initialize
        self._initialize(problem)
        self._iteration = 0
        self._convergence_history = []
        
        # Main optimization loop
        prev_fitness = float('inf')
        stagnant_iterations = 0
        max_stagnant = 50  # Stop nếu không improve sau 50 iterations
        
        while self._iteration < self.max_iterations:
            # Thực hiện một iteration
            current_best = self._iterate()
            self._iteration += 1
            
            # Track convergence
            self._convergence_history.append(current_best)
            
            # Check convergence
            if abs(prev_fitness - current_best) < self.tolerance:
                stagnant_iterations += 1
            else:
                stagnant_iterations = 0
            
            # Verbose output
            if self.verbose and self._iteration % 10 == 0:
                self._print_progress(current_best)
            
            # Early stopping
            if stagnant_iterations >= max_stagnant:
                if self.verbose:
                    print(f"\n[Early Stop] No improvement for {max_stagnant} iterations")
                break
            
            prev_fitness = current_best
        
        # Get final results
        best_solution, best_fitness = self._get_best_solution()
        execution_time = time.perf_counter() - self._start_time
        
        # Create results object
        results = OptimizationResults(
            solution=best_solution,
            fitness=best_fitness,
            success=True,  # Subclasses có thể override
            iterations=self._iteration,
            evaluations=self._evaluations,
            execution_time=execution_time,
            convergence_curve=self._convergence_history,
            metadata=self._get_metadata()
        )
        
        if self.verbose:
            self._print_final_results(results)
        
        return results
    
    def _print_progress(self, current_best: float) -> None:
        """In thông tin progress."""
        print(f"Iteration {self._iteration:4d} | "
              f"Best Fitness: {current_best:.6f} | "
              f"Evaluations: {self._evaluations}")
    
    def _print_final_results(self, results: OptimizationResults) -> None:
        """In kết quả cuối cùng."""
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
        """
        Trả về metadata cho results.
        Subclasses có thể override để thêm thông tin.
        """
        return {
            'algorithm': self.__class__.__name__,
            'max_iterations': self.max_iterations,
            'tolerance': self.tolerance,
            'seed': self.seed
        }
    
    def reset(self) -> None:
        """Reset algorithm state về ban đầu."""
        self._iteration = 0
        self._evaluations = 0
        self._best_fitness = float('inf')
        self._convergence_history = []
        if self.seed is not None:
            np.random.seed(self.seed)


class PopulationBasedOptimizer(BaseOptimizer):
    """
    Base class cho các thuật toán population-based (GA, PSO, ACO, etc.).
    
    Thêm các methods chung cho population management.
    """
    
    def __init__(self,
                 population_size: int = 50,
                 max_iterations: int = 1000,
                 tolerance: float = 1e-6,
                 verbose: bool = False,
                 seed: Optional[int] = None):
        """
        Args:
            population_size: Số lượng individuals trong population
            (các args khác như BaseOptimizer)
        """
        super().__init__(max_iterations, tolerance, verbose, seed)
        self.population_size = population_size
        self.population = None  # Will be initialized in subclasses
    
    def _initialize_random_population(self, 
                                     dimensions: int,
                                     lower_bound: float,
                                     upper_bound: float) -> np.ndarray:
        """
        Khởi tạo population ngẫu nhiên trong không gian continuous.
        
        Args:
            dimensions: Số dimensions của search space
            lower_bound: Giới hạn dưới
            upper_bound: Giới hạn trên
            
        Returns:
            population: Array shape (population_size, dimensions)
        """
        return np.random.uniform(
            lower_bound, 
            upper_bound, 
            size=(self.population_size, dimensions)
        )
    
    def _evaluate_population(self, 
                            population: np.ndarray,
                            objective_func) -> np.ndarray:
        """
        Evaluate fitness cho toàn bộ population.
        
        Args:
            population: Array shape (population_size, dimensions)
            objective_func: Function để evaluate fitness
            
        Returns:
            fitnesses: Array shape (population_size,)
        """
        fitnesses = np.array([objective_func(ind) for ind in population])
        self._evaluations += len(population)
        return fitnesses
    
    def _get_metadata(self) -> Dict[str, Any]:
        """Thêm population_size vào metadata."""
        metadata = super()._get_metadata()
        metadata['population_size'] = self.population_size
        return metadata


class SingleSolutionOptimizer(BaseOptimizer):
    """
    Base class cho các thuật toán single-solution based (Hill Climbing, SA, etc.).
    
    Thêm các methods chung cho single-solution management.
    """
    
    def __init__(self,
                 max_iterations: int = 1000,
                 tolerance: float = 1e-6,
                 verbose: bool = False,
                 seed: Optional[int] = None):
        super().__init__(max_iterations, tolerance, verbose, seed)
        self.current_solution = None
        self.current_fitness = float('inf')
    
    def _initialize_random_solution(self,
                                   dimensions: int,
                                   lower_bound: float,
                                   upper_bound: float) -> np.ndarray:
        """
        Khởi tạo solution ngẫu nhiên.
        
        Returns:
            solution: Array shape (dimensions,)
        """
        return np.random.uniform(lower_bound, upper_bound, size=dimensions)
    
    def _evaluate_solution(self, solution: np.ndarray, objective_func) -> float:
        """
        Evaluate fitness của một solution.
        
        Returns:
            fitness: Fitness value
        """
        self._evaluations += 1
        return objective_func(solution)