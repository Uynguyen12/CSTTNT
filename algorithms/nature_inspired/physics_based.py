"""
Physics-Based Algorithms
Gravitational Search Algorithm (GSA), Harmony Search (HS)
Note: Simulated Annealing đã có trong local_search.py
"""

import numpy as np
import math
from typing import Callable, Tuple, List, Optional, Any
from ..base_optimizer import PopulationBasedOptimizer, SingleSolutionOptimizer, OptimizationResults


class GravitationalSearchAlgorithm(PopulationBasedOptimizer):
    """
    Gravitational Search Algorithm (GSA)
    
    Đặc điểm:
        - Inspired by Newtonian law of gravity
        - Agents interact through gravitational force
        - Heavier agents attract lighter ones
    
    Physics laws:
        F = G * (M1 * M2) / (R + ε)
        a = F / M
        v(t+1) = rand * v(t) + a(t)
        x(t+1) = x(t) + v(t+1)
    
    Phù hợp cho:
        - Continuous optimization
        - Engineering problems
        - Function optimization
        - Global optimization
    """
    
    def __init__(self,
                 population_size: int = 50,
                 max_iterations: int = 1000,
                 bounds: Optional[List[Tuple[float, float]]] = None,
                 G0: float = 100.0,
                 alpha: float = 20.0,
                 tolerance: float = 1e-6,
                 verbose: bool = False,
                 seed: Optional[int] = None):
        """
        Args:
            population_size: Số lượng agents
            max_iterations: Số iterations tối đa
            bounds: Search space bounds - List of (lower, upper) tuples for each dimension
            G0: Initial gravitational constant
            alpha: Decay rate của G
            tolerance: Ngưỡng convergence
            verbose: In thông tin debug
            seed: Random seed
        """
        super().__init__(population_size, max_iterations, tolerance, verbose, seed)
        
        self.bounds = bounds
        self.G0 = G0
        self.alpha = alpha
        
        # Problem-specific attributes
        self.objective_func = None
        self.dimensions = None
        self.lower_bound = None
        self.upper_bound = None
        
        # GSA-specific attributes
        self.agents = None
        self.velocities = None
        self.fitness_values = None
        self.masses = None
        self.best_solution = None
        self.best_fitness = float('inf')
    
    def _initialize(self, problem: dict) -> None:
        """
        Khởi tạo GSA với problem definition.
        """
        self.objective_func = problem['objective_func']
        self.dimensions = problem['dimensions']
        self.lower_bound = problem.get('lower_bound', -100)
        self.upper_bound = problem.get('upper_bound', 100)
        
        # Initialize agents (positions)
        self.agents = self._initialize_random_population(
            self.dimensions,
            self.lower_bound,
            self.upper_bound
        )
        
        # Initialize velocities
        self.velocities = np.zeros((self.population_size, self.dimensions))
        
        # Evaluate
        self.fitness_values = self._evaluate_population(
            self.agents,
            self.objective_func
        )
        
        # Calculate initial masses
        self._update_masses()
        
        # Track best
        best_idx = np.argmin(self.fitness_values)
        self.best_solution = self.agents[best_idx].copy()
        self.best_fitness = self.fitness_values[best_idx]
    
    def _iterate(self) -> float:
        """
        Thực hiện một iteration của GSA.
        """
        # Update gravitational constant
        G = self.G0 * np.exp(-self.alpha * self._iteration / self.max_iterations)
        
        # Calculate forces and update velocities/positions
        for i in range(self.population_size):
            # Calculate total force on agent i
            force = np.zeros(self.dimensions)
            
            for j in range(self.population_size):
                if i != j:
                    # Euclidean distance
                    diff = self.agents[j] - self.agents[i]
                    distance = np.linalg.norm(diff) + 1e-10  # Avoid division by zero
                    
                    # Gravitational force
                    force += np.random.random() * self.masses[j] * diff / distance
            
            # Total force
            force = G * force
            
            # Acceleration (a = F / M_i)
            acceleration = force / (self.masses[i] + 1e-10)
            
            # Update velocity
            self.velocities[i] = np.random.random() * self.velocities[i] + acceleration
            
            # Update position
            self.agents[i] = self.agents[i] + self.velocities[i]
            
            # Boundary handling
            self.agents[i] = np.clip(
                self.agents[i],
                self.lower_bound,
                self.upper_bound
            )
        
        # Re-evaluate
        self.fitness_values = self._evaluate_population(
            self.agents,
            self.objective_func
        )
        
        # Update masses
        self._update_masses()
        
        # Update best
        best_idx = np.argmin(self.fitness_values)
        if self.fitness_values[best_idx] < self.best_fitness:
            self.best_solution = self.agents[best_idx].copy()
            self.best_fitness = self.fitness_values[best_idx]
        self.position_history.append(self.agents.copy())
        self.best_history.append(self.best_solution.copy())

        return self.best_fitness
    
    def _update_masses(self) -> None:
        """
        Cập nhật masses dựa trên fitness values.
        Better fitness = heavier mass
        """
        # Normalize fitness to [0, 1]
        best = np.min(self.fitness_values)
        worst = np.max(self.fitness_values)
        
        if worst - best < 1e-10:
            # All agents have similar fitness
            self.masses = np.ones(self.population_size) / self.population_size
        else:
            # Calculate masses (better fitness = higher mass)
            normalized_fitness = (self.fitness_values - worst) / (best - worst)
            
            # Mass calculation
            self.masses = normalized_fitness / np.sum(normalized_fitness)
    
    def _get_best_solution(self) -> Tuple[np.ndarray, float]:
        """Trả về best solution."""
        return self.best_solution.copy(), self.best_fitness
    
    def _get_metadata(self) -> dict:
        """Thêm GSA-specific metadata."""
        metadata = super()._get_metadata()
        metadata.update({
            'G0': self.G0,
            'alpha': self.alpha
        })
        return metadata


class HarmonySearch(PopulationBasedOptimizer):
    """
    Harmony Search (HS) Algorithm
    
    Đặc điểm:
        - Inspired by musical improvisation
        - Musicians improvise to find harmony
        - Balance between memory and randomization
    
    Operations:
        1. Memory consideration (HMCR)
        2. Pitch adjustment (PAR)
        3. Random selection
    
    Phù hợp cho:
        - Continuous optimization
        - Parameter tuning
        - Engineering design
        - Structural optimization
    """
    
    def __init__(self,
                 harmony_memory_size: int = 50,
                 max_iterations: int = 1000,
                 bounds: Optional[List[Tuple[float, float]]] = None,
                 hmcr: float = 0.9,
                 par: float = 0.3,
                 bandwidth: float = 0.01,
                 tolerance: float = 1e-6,
                 verbose: bool = False,
                 seed: Optional[int] = None):
        """
        Args:
            harmony_memory_size: Kích thước harmony memory (HMS)
            max_iterations: Số iterations tối đa
            bounds: Search space bounds - List of (lower, upper) tuples for each dimension
            hmcr: Harmony Memory Considering Rate (0-1)
            par: Pitch Adjustment Rate (0-1)
            bandwidth: Bandwidth for pitch adjustment
            tolerance: Ngưỡng convergence
            verbose: In thông tin debug
            seed: Random seed
        """
        super().__init__(harmony_memory_size, max_iterations, tolerance, verbose, seed)
        
        self.bounds = bounds
        self.hmcr = hmcr
        self.par = par
        self.bandwidth = bandwidth
        
        # Problem-specific attributes
        self.objective_func = None
        self.dimensions = None
        self.lower_bound = None
        self.upper_bound = None
        
        # HS-specific attributes
        self.harmony_memory = None
        self.fitness_values = None
        self.best_harmony = None
        self.best_fitness = float('inf')
    
    def _initialize(self, problem: dict) -> None:
        """
        Khởi tạo HS với problem definition.
        """
        self.objective_func = problem['objective_func']
        self.dimensions = problem['dimensions']
        self.lower_bound = problem.get('lower_bound', -100)
        self.upper_bound = problem.get('upper_bound', 100)
        
        # Initialize harmony memory
        self.harmony_memory = self._initialize_random_population(
            self.dimensions,
            self.lower_bound,
            self.upper_bound
        )
        
        # Evaluate
        self.fitness_values = self._evaluate_population(
            self.harmony_memory,
            self.objective_func
        )
        
        # Track best
        best_idx = np.argmin(self.fitness_values)
        self.best_harmony = self.harmony_memory[best_idx].copy()
        self.best_fitness = self.fitness_values[best_idx]
    
    def _iterate(self) -> float:
        """
        Thực hiện một iteration của HS.
        """
        # Generate new harmony
        new_harmony = self._improvise_new_harmony()
        
        # Evaluate
        new_fitness = self.objective_func(new_harmony)
        self._evaluations += 1
        
        # Update harmony memory if better than worst
        worst_idx = np.argmax(self.fitness_values)
        if new_fitness < self.fitness_values[worst_idx]:
            self.harmony_memory[worst_idx] = new_harmony
            self.fitness_values[worst_idx] = new_fitness
            
            # Update best
            if new_fitness < self.best_fitness:
                self.best_harmony = new_harmony.copy()
                self.best_fitness = new_fitness
            self.position_history.append(self.harmony_memory.copy())
            self.best_history.append(self.best_harmony.copy())

        return self.best_fitness
    
    def _improvise_new_harmony(self) -> np.ndarray:
        """
        Improvise new harmony vector.
        
        Returns:
            new_harmony: New candidate solution
        """
        new_harmony = np.zeros(self.dimensions)
        
        for i in range(self.dimensions):
            if np.random.random() < self.hmcr:
                # Memory consideration
                # Randomly select from harmony memory
                selected_idx = np.random.randint(0, self.population_size)
                new_harmony[i] = self.harmony_memory[selected_idx, i]
                
                # Pitch adjustment
                if np.random.random() < self.par:
                    # Adjust pitch
                    adjustment = self.bandwidth * (self.upper_bound - self.lower_bound)
                    new_harmony[i] += np.random.uniform(-adjustment, adjustment)
            else:
                # Random selection
                new_harmony[i] = np.random.uniform(
                    self.lower_bound,
                    self.upper_bound
                )
            
            # Boundary handling
            new_harmony[i] = np.clip(
                new_harmony[i],
                self.lower_bound,
                self.upper_bound
            )
        
        return new_harmony
    
    def _get_best_solution(self) -> Tuple[np.ndarray, float]:
        """Trả về best solution."""
        return self.best_harmony.copy(), self.best_fitness
    
    def _get_metadata(self) -> dict:
        """Thêm HS-specific metadata."""
        metadata = super()._get_metadata()
        metadata.update({
            'hmcr': self.hmcr,
            'par': self.par,
            'bandwidth': self.bandwidth
        })
        return metadata


class SimulatedAnnealing(SingleSolutionOptimizer):
    """
    Simulated Annealing (SA)
    
    Đặc điểm:
        - Inspired by annealing in metallurgy (cooling process)
        - Probabilistic technique for approximating global optimum
        - Can escape local optima via probabilistic acceptance
    
    Strategy:
        - Start with high temperature (accept bad solutions)
        - Gradually decrease temperature (less accepting)
        - Accept worse solutions with decreasing probability
    
    Phù hợp cho:
        - Combinatorial optimization
        - Non-convex continuous optimization
        - Escaping local optima
    """
    
    def __init__(self,
                 bounds: Optional[List[Tuple[float, float]]] = None,
                 max_iterations: int = 1000,
                 initial_temperature: float = 100.0,
                 cooling_rate: float = 0.95,
                 temperature_min: float = 0.01,
                 perturbation_scale: float = 0.5,
                 tolerance: float = 1e-6,
                 verbose: bool = False,
                 seed: Optional[int] = None):
        """
        Args:
            bounds: Search space bounds
            max_iterations: Số iterations tối đa
            initial_temperature: Nhiệt độ ban đầu
            cooling_rate: Tỷ lệ giảm nhiệt độ
            temperature_min: Nhiệt độ tối thiểu
            perturbation_scale: Scale của perturbation
            tolerance: Threshold convergence
            verbose: In thông tin debug
            seed: Random seed
        """
        super().__init__(max_iterations, tolerance, verbose, seed)
        
        self.bounds = bounds
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.temperature_min = temperature_min
        self.perturbation_scale = perturbation_scale
        
        # Problem-specific
        self.objective_func = None
        self.dimensions = None
        self.lower_bound = None
        self.upper_bound = None
        
        # SA-specific
        self.current_solution = None
        self.current_fitness = float('inf')
        self.best_solution = None
        self.best_fitness = float('inf')
    
    def _initialize(self, problem: Any) -> None:
        """Initialize SA with problem."""
        from ..base_optimizer import Any
        self.objective_func = problem if callable(problem) else problem.get('objective_func')
        
        # Setup bounds
        if self.bounds is None:
            self.bounds = [(-5.0, 5.0)] * 3  # Default 3D
        
        self.dimensions = len(self.bounds)
        self.lower_bound = np.array([b[0] for b in self.bounds])
        self.upper_bound = np.array([b[1] for b in self.bounds])
        
        # Initialize solution
        self.current_solution = self._initialize_random_solution(
            self.dimensions, self.lower_bound, self.upper_bound
        )
        self.current_fitness = self._evaluate_solution(self.current_solution, self.objective_func)
        self.best_solution = self.current_solution.copy()
        self.best_fitness = self.current_fitness
    
    def _iterate(self) -> float:
        """One iteration of SA."""
        temperature = self.initial_temperature * (self.cooling_rate ** self._iteration)
        temperature = max(temperature, self.temperature_min)
        
        # Perturbation
        neighbor = self.current_solution + np.random.normal(0, self.perturbation_scale, self.dimensions)
        neighbor = np.clip(neighbor, self.lower_bound, self.upper_bound)
        
        neighbor_fitness = self._evaluate_solution(neighbor, self.objective_func)
        
        # Acceptance
        if neighbor_fitness < self.current_fitness:
            self.current_solution = neighbor
            self.current_fitness = neighbor_fitness
        else:
            delta = neighbor_fitness - self.current_fitness
            if temperature > 1e-10 and np.random.random() < np.exp(-delta / temperature):
                self.current_solution = neighbor
                self.current_fitness = neighbor_fitness
        
        # Track best
        if self.current_fitness < self.best_fitness:
            self.best_solution = self.current_solution.copy()
            self.best_fitness = self.current_fitness
        
        return self.best_fitness
    
    def _get_best_solution(self) -> Tuple[Any, float]:
        """Return best solution found."""
        return self.best_solution, self.best_fitness
    
    def _initialize_random_solution(self, dimensions, lower_bound, upper_bound):
        """Initialize random solution in bounds."""
        return np.random.uniform(lower_bound, upper_bound, dimensions)
    
    def _evaluate_solution(self, solution, objective_func):
        """Evaluate fitness."""
        self._evaluations += 1
        return objective_func(solution)


class HillClimbing(SingleSolutionOptimizer):
    """
    Hill Climbing (Greedy Local Search)
    
    Đặc điểm:
        - Simple greedy approach
        - Always move to better neighbor
        - Can get stuck in local optima
        - Fast convergence
    
    Strategy:
        - Start with random solution
        - Always move to better neighbor
        - Stop when no improvement found
    
    Phù hợp cho:
        - Local optimization
        - Simple problems
        - When global optimum not needed
    """
    
    def __init__(self,
                 bounds: Optional[List[Tuple[float, float]]] = None,
                 max_iterations: int = 1000,
                 step_size: float = 0.5,
                 step_size_reduction: float = 0.99,
                 step_size_min: float = 0.001,
                 tolerance: float = 1e-6,
                 verbose: bool = False,
                 seed: Optional[int] = None):
        """
        Args:
            bounds: Search space bounds
            max_iterations: Số iterations tối đa
            step_size: Kích thước bước
            step_size_reduction: Tỷ lệ giảm step size
            step_size_min: Step size tối thiểu
            tolerance: Threshold convergence
            verbose: In thông tin debug
            seed: Random seed
        """
        super().__init__(max_iterations, tolerance, verbose, seed)
        
        self.bounds = bounds
        self.step_size = step_size
        self.step_size_reduction = step_size_reduction
        self.step_size_min = step_size_min
        
        # Problem-specific
        self.objective_func = None
        self.dimensions = None
        self.lower_bound = None
        self.upper_bound = None
        
        # HC-specific
        self.current_solution = None
        self.current_fitness = float('inf')
        self.best_solution = None
        self.best_fitness = float('inf')
    
    def _initialize(self, problem: Any) -> None:
        """Initialize HC with problem."""
        from ..base_optimizer import Any
        self.objective_func = problem if callable(problem) else problem.get('objective_func')
        
        # Setup bounds
        if self.bounds is None:
            self.bounds = [(-5.0, 5.0)] * 3
        
        self.dimensions = len(self.bounds)
        self.lower_bound = np.array([b[0] for b in self.bounds])
        self.upper_bound = np.array([b[1] for b in self.bounds])
        
        # Initialize solution
        self.current_solution = self._initialize_random_solution(
            self.dimensions, self.lower_bound, self.upper_bound
        )
        self.current_fitness = self._evaluate_solution(self.current_solution, self.objective_func)
        self.best_solution = self.current_solution.copy()
        self.best_fitness = self.current_fitness
    
    def _iterate(self) -> float:
        """One iteration of HC."""
        improved = False
        current_step = max(self.step_size * (self.step_size_reduction ** self._iteration), self.step_size_min)
        
        # Try random directions
        for _ in range(self.dimensions):
            neighbor = self.current_solution.copy()
            direction = np.random.randn(self.dimensions)
            direction = direction / np.linalg.norm(direction)
            neighbor = neighbor + current_step * direction
            neighbor = np.clip(neighbor, self.lower_bound, self.upper_bound)
            
            neighbor_fitness = self._evaluate_solution(neighbor, self.objective_func)
            
            if neighbor_fitness < self.current_fitness:
                self.current_solution = neighbor
                self.current_fitness = neighbor_fitness
                improved = True
                
                if self.current_fitness < self.best_fitness:
                    self.best_solution = self.current_solution.copy()
                    self.best_fitness = self.current_fitness
                
                break  # Move immediately to better solution
        
        if not improved:
            # No improvement found, step size decreases automatically
            pass
        
        return self.best_fitness
    
    def _get_best_solution(self) -> Tuple[Any, float]:
        """Return best solution found."""
        return self.best_solution, self.best_fitness
    
    def _initialize_random_solution(self, dimensions, lower_bound, upper_bound):
        """Initialize random solution in bounds."""
        return np.random.uniform(lower_bound, upper_bound, dimensions)
    
    def _evaluate_solution(self, solution, objective_func):
        """Evaluate fitness."""
        self._evaluations += 1
        return objective_func(solution)