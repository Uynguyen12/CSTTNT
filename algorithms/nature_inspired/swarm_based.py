"""
Swarm Intelligence / Biology-Based Algorithms
PSO, ACO, ABC, FA, CS
"""

import numpy as np
from typing import Callable, Tuple, List, Optional, Dict
from ..base_optimizer import PopulationBasedOptimizer, OptimizationResults


class ParticleSwarmOptimization(PopulationBasedOptimizer):
    """
    Particle Swarm Optimization (PSO)
    
    Đặc điểm:
        - Inspired by bird flocking and fish schooling
        - Particles move through search space
        - Balance between exploration and exploitation
    
    Update equations:
        v_i(t+1) = w*v_i(t) + c1*r1*(pbest_i - x_i(t)) + c2*r2*(gbest - x_i(t))
        x_i(t+1) = x_i(t) + v_i(t+1)
    
    Phù hợp cho:
        - Continuous optimization
        - Hyperparameter tuning
        - Engineering design
        - Function optimization
    """
    
    def __init__(self,
                 population_size: int = 50,
                 max_iterations: int = 1000,
                 bounds: Optional[List[Tuple[float, float]]] = None,
                 inertia_weight: float = 0.7,
                 cognitive_coeff: float = 1.5,
                 social_coeff: float = 1.5,
                 max_velocity: Optional[float] = None,
                 tolerance: float = 1e-6,
                 verbose: bool = False,
                 seed: Optional[int] = None):
        """
        Args:
            population_size: Số lượng particles
            max_iterations: Số iterations tối đa
            bounds: Search space bounds - List of (lower, upper) tuples for each dimension
            inertia_weight (w): Trọng số quán tính (0.4 - 0.9)
            cognitive_coeff (c1): Hệ số nhận thức (learning from own experience)
            social_coeff (c2): Hệ số xã hội (learning from swarm)
            max_velocity: Giới hạn vận tốc (None = không giới hạn)
            tolerance: Ngưỡng convergence
            verbose: In thông tin debug
            seed: Random seed
        """
        super().__init__(population_size, max_iterations, tolerance, verbose, seed)
        
        self.bounds = bounds
        self.inertia_weight = inertia_weight
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff
        self.max_velocity = max_velocity
        
        # Problem-specific attributes
        self.objective_func = None
        self.dimensions = None
        self.lower_bound = None
        self.upper_bound = None
        
        # PSO-specific attributes
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_fitness = None
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.fitness_values = None
    
    def _initialize(self, problem: dict) -> None:
        """
        Khởi tạo PSO với problem definition.
        """
        self.objective_func = problem['objective_func']
        self.dimensions = problem['dimensions']
        self.lower_bound = problem.get('lower_bound', -100)
        self.upper_bound = problem.get('upper_bound', 100)
        
        # Initialize particles (positions)
        self.population = self._initialize_random_population(
            self.dimensions,
            self.lower_bound,
            self.upper_bound
        )
        
        # Initialize velocities
        velocity_range = (self.upper_bound - self.lower_bound) * 0.1
        self.velocities = np.random.uniform(
            -velocity_range,
            velocity_range,
            size=(self.population_size, self.dimensions)
        )
        
        # Set max velocity if not specified
        if self.max_velocity is None:
            self.max_velocity = (self.upper_bound - self.lower_bound) * 0.2
        
        # Evaluate initial population
        self.fitness_values = self._evaluate_population(
            self.population,
            self.objective_func
        )
        
        # Initialize personal best
        self.personal_best_positions = self.population.copy()
        self.personal_best_fitness = self.fitness_values.copy()
        
        # Initialize global best
        best_idx = np.argmin(self.fitness_values)
        self.global_best_position = self.population[best_idx].copy()
        self.global_best_fitness = self.fitness_values[best_idx]
        
    
    def _iterate(self) -> float:
        """
        Thực hiện một iteration của PSO.
        """
        # Update velocities and positions
        for i in range(self.population_size):
            # Random factors
            r1 = np.random.random(self.dimensions)
            r2 = np.random.random(self.dimensions)
            
            # Cognitive component
            cognitive = self.cognitive_coeff * r1 * (
                self.personal_best_positions[i] - self.population[i]
            )
            
            # Social component
            social = self.social_coeff * r2 * (
                self.global_best_position - self.population[i]
            )
            
            # Update velocity
            self.velocities[i] = (
                self.inertia_weight * self.velocities[i] +
                cognitive + social
            )
            
            # Limit velocity
            self.velocities[i] = np.clip(
                self.velocities[i],
                -self.max_velocity,
                self.max_velocity
            )
            
            # Update position
            self.population[i] = self.population[i] + self.velocities[i]
            
            # Boundary handling
            self.population[i] = np.clip(
                self.population[i],
                self.lower_bound,
                self.upper_bound
            )
        
        # Evaluate new positions
        self.fitness_values = self._evaluate_population(
            self.population,
            self.objective_func
        )
        
        # Update personal best
        improved = self.fitness_values < self.personal_best_fitness
        self.personal_best_positions[improved] = self.population[improved].copy()
        self.personal_best_fitness[improved] = self.fitness_values[improved]
        
        # Update global best
        best_idx = np.argmin(self.fitness_values)
        if self.fitness_values[best_idx] < self.global_best_fitness:
            self.global_best_position = self.population[best_idx].copy()
            self.global_best_fitness = self.fitness_values[best_idx]
        
        self.position_history.append(self.population.copy())

        return self.global_best_fitness
    
    def _get_best_solution(self) -> Tuple[np.ndarray, float]:
        """Trả về best solution."""
        return self.global_best_position.copy(), self.global_best_fitness
    
    def _get_metadata(self) -> dict:
        """Thêm PSO-specific metadata."""
        metadata = super()._get_metadata()
        metadata.update({
            'inertia_weight': self.inertia_weight,
            'cognitive_coeff': self.cognitive_coeff,
            'social_coeff': self.social_coeff,
            'max_velocity': self.max_velocity
        })
        return metadata


class AntColonyOptimization(PopulationBasedOptimizer):
    """
    Ant Colony Optimization (ACO)
    
    Đặc điểm:
        - Inspired by foraging behavior of ants
        - Uses pheromone trails for communication
        - Probabilistic construction of solutions
    
    Phù hợp cho:
        - Traveling Salesman Problem (TSP)
        - Routing problems
        - Scheduling
        - Combinatorial optimization
    
    Note: Implement này focused on continuous optimization variant (ACOR)
    """
    
    def __init__(self,
                 population_size: int = 50,
                 max_iterations: int = 1000,
                 archive_size: int = 50,
                 intensification_factor: float = 0.85,
                 deviation_distance: float = 0.1,
                 tolerance: float = 1e-6,
                 verbose: bool = False,
                 seed: Optional[int] = None):
        """
        Args:
            population_size: Số lượng ants
            max_iterations: Số iterations tối đa
            archive_size: Kích thước solution archive
            intensification_factor (q): Điều khiển intensification (0-1)
            deviation_distance (ξ): Standard deviation cho Gaussian kernel
            tolerance: Ngưỡng convergence
            verbose: In thông tin debug
            seed: Random seed
        """
        super().__init__(population_size, max_iterations, tolerance, verbose, seed)
        
        self.archive_size = archive_size
        self.intensification_factor = intensification_factor
        self.deviation_distance = deviation_distance
        
        # Problem-specific attributes
        self.objective_func = None
        self.dimensions = None
        self.lower_bound = None
        self.upper_bound = None
        
        # ACO-specific attributes
        self.archive = None  # Solution archive
        self.archive_fitness = None
        self.weights = None  # Selection weights
        self.best_solution = None
        self.best_fitness = float('inf')
    
    def _initialize(self, problem: dict) -> None:
        """
        Khởi tạo ACO với problem definition.
        """
        self.objective_func = problem['objective_func']
        self.dimensions = problem['dimensions']
        self.lower_bound = problem.get('lower_bound', -100)
        self.upper_bound = problem.get('upper_bound', 100)
        
        # Initialize archive with random solutions
        self.archive = self._initialize_random_population(
            self.dimensions,
            self.lower_bound,
            self.upper_bound
        )
        
        # Limit archive size
        if len(self.archive) > self.archive_size:
            self.archive = self.archive[:self.archive_size]
        
        # Evaluate archive
        self.archive_fitness = self._evaluate_population(
            self.archive,
            self.objective_func
        )
        
        # Sort archive by fitness
        sorted_indices = np.argsort(self.archive_fitness)
        self.archive = self.archive[sorted_indices]
        self.archive_fitness = self.archive_fitness[sorted_indices]
        
        # Calculate weights (using Gaussian function)
        self._update_weights()
        
        # Track best
        self.best_solution = self.archive[0].copy()
        self.best_fitness = self.archive_fitness[0]
    
    def _update_weights(self) -> None:
        """
        Cập nhật selection weights cho archive solutions.
        Sử dụng Gaussian function.
        """
        k = len(self.archive)
        q = self.intensification_factor
        
        weights = np.zeros(k)
        for i in range(k):
            weights[i] = (1.0 / (q * k * np.sqrt(2 * np.pi))) * np.exp(
                -(i ** 2) / (2 * (q * k) ** 2)
            )
        
        # Normalize
        self.weights = weights / np.sum(weights)
    
    def _iterate(self) -> float:
        """
        Thực hiện một iteration của ACO.
        """
        new_solutions = []
        new_fitness = []
        
        # Generate new solutions
        for _ in range(self.population_size):
            # Select solution from archive based on weights
            selected_idx = np.random.choice(len(self.archive), p=self.weights)
            selected_solution = self.archive[selected_idx]
            
            # Generate new solution using Gaussian kernel
            new_solution = np.zeros(self.dimensions)
            for d in range(self.dimensions):
                # Calculate standard deviation
                distances = np.abs(self.archive[:, d] - selected_solution[d])
                sigma = self.deviation_distance * np.sum(
                    self.weights * distances
                )
                
                # Sample from Gaussian
                new_solution[d] = np.random.normal(
                    selected_solution[d],
                    sigma
                )
            
            # Boundary handling
            new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
            
            # Evaluate
            fitness = self.objective_func(new_solution)
            self._evaluations += 1
            
            new_solutions.append(new_solution)
            new_fitness.append(fitness)
        
        # Combine with archive
        combined_solutions = np.vstack([self.archive, new_solutions])
        combined_fitness = np.concatenate([self.archive_fitness, new_fitness])
        
        # Sort and keep best archive_size solutions
        sorted_indices = np.argsort(combined_fitness)[:self.archive_size]
        self.archive = combined_solutions[sorted_indices]
        self.archive_fitness = combined_fitness[sorted_indices]
        
        # Update weights
        self._update_weights()
        
        # Update best
        if self.archive_fitness[0] < self.best_fitness:
            self.best_solution = self.archive[0].copy()
            self.best_fitness = self.archive_fitness[0]

        return self.best_fitness
    
    def _get_best_solution(self) -> Tuple[np.ndarray, float]:
        """Trả về best solution."""
        return self.best_solution.copy(), self.best_fitness
    
    def _get_metadata(self) -> dict:
        """Thêm ACO-specific metadata."""
        metadata = super()._get_metadata()
        metadata.update({
            'archive_size': self.archive_size,
            'intensification_factor': self.intensification_factor,
            'deviation_distance': self.deviation_distance
        })
        return metadata


class ArtificialBeeColony(PopulationBasedOptimizer):
    """
    Artificial Bee Colony (ABC) Algorithm
    
    Đặc điểm:
        - Inspired by foraging behavior of honey bees
        - Three types of bees: employed, onlooker, scout
        - Balance exploration and exploitation
    
    Phases:
        1. Employed bee phase: Local search around food sources
        2. Onlooker bee phase: Select promising sources
        3. Scout bee phase: Replace abandoned sources
    
    Phù hợp cho:
        - Continuous optimization
        - Engineering design
        - Parameter optimization
        - Large-scale problems
    """
    
    def __init__(self,
                 population_size: int = 50,
                 max_iterations: int = 1000,
                 limit: Optional[int] = None,
                 tolerance: float = 1e-6,
                 verbose: bool = False,
                 seed: Optional[int] = None):
        """
        Args:
            population_size: Số lượng food sources (employed bees)
            max_iterations: Số iterations tối đa
            limit: Số trials trước khi abandon food source (default: population_size * dimensions)
            tolerance: Ngưỡng convergence
            verbose: In thông tin debug
            seed: Random seed
        """
        super().__init__(population_size, max_iterations, tolerance, verbose, seed)
        
        self.limit = limit
        
        # Problem-specific attributes
        self.objective_func = None
        self.dimensions = None
        self.lower_bound = None
        self.upper_bound = None
        
        # ABC-specific attributes
        self.food_sources = None  # Population of food sources
        self.fitness_values = None
        self.trial_counter = None  # Number of trials for each source
        self.best_solution = None
        self.best_fitness = float('inf')
    
    def _initialize(self, problem: dict) -> None:
        """
        Khởi tạo ABC với problem definition.
        """
        self.objective_func = problem['objective_func']
        self.dimensions = problem['dimensions']
        self.lower_bound = problem.get('lower_bound', -100)
        self.upper_bound = problem.get('upper_bound', 100)
        
        # Set limit if not specified
        if self.limit is None:
            self.limit = self.population_size * self.dimensions
        
        # Initialize food sources
        self.food_sources = self._initialize_random_population(
            self.dimensions,
            self.lower_bound,
            self.upper_bound
        )
        
        # Evaluate
        self.fitness_values = self._evaluate_population(
            self.food_sources,
            self.objective_func
        )
        
        # Initialize trial counters
        self.trial_counter = np.zeros(self.population_size, dtype=int)
        
        # Track best
        best_idx = np.argmin(self.fitness_values)
        self.best_solution = self.food_sources[best_idx].copy()
        self.best_fitness = self.fitness_values[best_idx]
    
    def _iterate(self) -> float:
        """
        Thực hiện một iteration của ABC.
        """
        # Employed bee phase
        self._employed_bee_phase()
        
        # Onlooker bee phase
        self._onlooker_bee_phase()
        
        # Scout bee phase
        self._scout_bee_phase()
        
        # Update best
        best_idx = np.argmin(self.fitness_values)
        if self.fitness_values[best_idx] < self.best_fitness:
            self.best_solution = self.food_sources[best_idx].copy()
            self.best_fitness = self.fitness_values[best_idx]
        self.position_history.append(self.food_sources.copy())
        

        return self.best_fitness
    
    def _employed_bee_phase(self) -> None:
        """
        Employed bees search around their food sources.
        """
        for i in range(self.population_size):
            # Generate new candidate solution
            new_solution = self._generate_neighbor(i)
            
            # Evaluate
            new_fitness = self.objective_func(new_solution)
            self._evaluations += 1
            
            # Greedy selection
            if new_fitness < self.fitness_values[i]:
                self.food_sources[i] = new_solution
                self.fitness_values[i] = new_fitness
                self.trial_counter[i] = 0
            else:
                self.trial_counter[i] += 1
    
    def _onlooker_bee_phase(self) -> None:
        """
        Onlooker bees select food sources based on fitness.
        """
        # Calculate selection probabilities (fitness-based)
        # Convert to maximization: higher fitness = better
        max_fitness = np.max(self.fitness_values)
        adjusted_fitness = max_fitness - self.fitness_values + 1e-10
        probabilities = adjusted_fitness / np.sum(adjusted_fitness)
        
        for _ in range(self.population_size):
            # Select food source based on probability
            i = np.random.choice(self.population_size, p=probabilities)
            
            # Generate new candidate
            new_solution = self._generate_neighbor(i)
            
            # Evaluate
            new_fitness = self.objective_func(new_solution)
            self._evaluations += 1
            
            # Greedy selection
            if new_fitness < self.fitness_values[i]:
                self.food_sources[i] = new_solution
                self.fitness_values[i] = new_fitness
                self.trial_counter[i] = 0
            else:
                self.trial_counter[i] += 1
    
    def _scout_bee_phase(self) -> None:
        """
        Scout bees abandon exhausted food sources.
        """
        # Find sources that exceeded limit
        abandoned = self.trial_counter > self.limit
        
        if np.any(abandoned):
            for i in np.where(abandoned)[0]:
                # Generate new random food source
                self.food_sources[i] = np.random.uniform(
                    self.lower_bound,
                    self.upper_bound,
                    size=self.dimensions
                )
                
                # Evaluate
                self.fitness_values[i] = self.objective_func(self.food_sources[i])
                self._evaluations += 1
                
                # Reset counter
                self.trial_counter[i] = 0
    
    def _generate_neighbor(self, i: int) -> np.ndarray:
        """
        Generate neighbor solution for food source i.
        
        Formula: v_i = x_i + phi * (x_i - x_k)
        where k is random source != i, phi is random in [-1, 1]
        """
        # Select random dimension
        j = np.random.randint(0, self.dimensions)
        
        # Select random source (different from i)
        k = i
        while k == i:
            k = np.random.randint(0, self.population_size)
        
        # Generate new solution
        new_solution = self.food_sources[i].copy()
        phi = np.random.uniform(-1, 1)
        new_solution[j] = self.food_sources[i][j] + phi * (
            self.food_sources[i][j] - self.food_sources[k][j]
        )
        
        # Boundary handling
        new_solution[j] = np.clip(
            new_solution[j],
            self.lower_bound,
            self.upper_bound
        )
        
        return new_solution
    
    def _get_best_solution(self) -> Tuple[np.ndarray, float]:
        """Trả về best solution."""
        return self.best_solution.copy(), self.best_fitness
    
    def _get_metadata(self) -> dict:
        """Thêm ABC-specific metadata."""
        metadata = super()._get_metadata()
        metadata.update({
            'limit': self.limit
        })
        return metadata


class FireflyAlgorithm(PopulationBasedOptimizer):
    """
    Firefly Algorithm (FA)
    
    Đặc điểm:
        - Inspired by flashing behavior of fireflies
        - Attraction based on brightness (fitness)
        - Light intensity decreases with distance
    
    Movement equation:
        x_i = x_i + β₀*e^(-γr²)*(x_j - x_i) + α*ε
        
    Phù hợp cho:
        - Multimodal optimization
        - Engineering design
        - Image processing
        - Continuous optimization
    """
    
    def __init__(self,
                 population_size: int = 50,
                 max_iterations: int = 1000,
                 alpha: float = 0.5,
                 beta_min: float = 0.2,
                 gamma: float = 1.0,
                 tolerance: float = 1e-6,
                 verbose: bool = False,
                 seed: Optional[int] = None):
        """
        Args:
            population_size: Số lượng fireflies
            max_iterations: Số iterations tối đa
            alpha: Randomization parameter (0-1)
            beta_min: Minimum attractiveness (0-1)
            gamma: Light absorption coefficient
            tolerance: Ngưỡng convergence
            verbose: In thông tin debug
            seed: Random seed
        """
        super().__init__(population_size, max_iterations, tolerance, verbose, seed)
        
        self.alpha = alpha
        self.beta_min = beta_min
        self.gamma = gamma
        
        # Problem-specific attributes
        self.objective_func = None
        self.dimensions = None
        self.lower_bound = None
        self.upper_bound = None
        
        # FA-specific attributes
        self.fireflies = None
        self.light_intensity = None  # Fitness values
        self.best_solution = None
        self.best_fitness = float('inf')
    
    def _initialize(self, problem: dict) -> None:
        """
        Khởi tạo FA với problem definition.
        """
        self.objective_func = problem['objective_func']
        self.dimensions = problem['dimensions']
        self.lower_bound = problem.get('lower_bound', -100)
        self.upper_bound = problem.get('upper_bound', 100)
        
        # Initialize fireflies
        self.fireflies = self._initialize_random_population(
            self.dimensions,
            self.lower_bound,
            self.upper_bound
        )
        
        # Evaluate (light intensity = inverse of fitness for minimization)
        fitness_values = self._evaluate_population(
            self.fireflies,
            self.objective_func
        )
        self.light_intensity = 1.0 / (fitness_values + 1e-10)
        
        # Track best
        best_idx = np.argmax(self.light_intensity)
        self.best_solution = self.fireflies[best_idx].copy()
        self.best_fitness = fitness_values[best_idx]
    
    def _iterate(self) -> float:
        """
        Thực hiện một iteration của FA.
        """
        # Scale factor for search space
        scale = self.upper_bound - self.lower_bound
        
        # Move fireflies
        for i in range(self.population_size):
            for j in range(self.population_size):
                # Move firefly i towards j if j is brighter
                if self.light_intensity[j] > self.light_intensity[i]:
                    # Calculate distance
                    r = np.linalg.norm(self.fireflies[i] - self.fireflies[j])
                    
                    # Calculate attractiveness
                    beta = self.beta_min + (1 - self.beta_min) * np.exp(-self.gamma * r * r)
                    
                    # Update position
                    random_component = self.alpha * (np.random.random(self.dimensions) - 0.5) * scale
                    
                    self.fireflies[i] = self.fireflies[i] + \
                        beta * (self.fireflies[j] - self.fireflies[i]) + \
                        random_component
                    
                    # Boundary handling
                    self.fireflies[i] = np.clip(
                        self.fireflies[i],
                        self.lower_bound,
                        self.upper_bound
                    )
        
        # Re-evaluate all fireflies
        fitness_values = self._evaluate_population(
            self.fireflies,
            self.objective_func
        )
        self.light_intensity = 1.0 / (fitness_values + 1e-10)
        
        # Update best
        best_idx = np.argmax(self.light_intensity)
        if fitness_values[best_idx] < self.best_fitness:
            self.best_solution = self.fireflies[best_idx].copy()
            self.best_fitness = fitness_values[best_idx]
        self.position_history.append(self.fireflies.copy())

        # Decrease randomization parameter
        self.alpha *= 0.97
        
        return self.best_fitness
    
    def _get_best_solution(self) -> Tuple[np.ndarray, float]:
        """Trả về best solution."""
        return self.best_solution.copy(), self.best_fitness
    
    def _get_metadata(self) -> dict:
        """Thêm FA-specific metadata."""
        metadata = super()._get_metadata()
        metadata.update({
            'alpha': self.alpha,
            'beta_min': self.beta_min,
            'gamma': self.gamma
        })
        return metadata


class CuckooSearch(PopulationBasedOptimizer):
    """
    Cuckoo Search (CS) Algorithm
    
    Đặc điểm:
        - Inspired by brood parasitism of cuckoo species
        - Uses Lévy flights for exploration
        - Abandons worst nests with probability pa
    
    Lévy flight:
        Step size follows Lévy distribution (heavy-tailed)
        Good for both local and global search
    
    Phù hợp cho:
        - Continuous optimization
        - Engineering design
        - Feature selection
        - Scheduling
    """
    
    def __init__(self,
                 population_size: int = 50,
                 max_iterations: int = 1000,
                 pa: float = 0.25,
                 alpha: float = 0.01,
                 beta: float = 1.5,
                 tolerance: float = 1e-6,
                 verbose: bool = False,
                 seed: Optional[int] = None):
        """
        Args:
            population_size: Số lượng nests
            max_iterations: Số iterations tối đa
            pa: Probability of abandoning worst nests (0-1)
            alpha: Step size scaling factor
            beta: Lévy exponent (typically 1.5)
            tolerance: Ngưỡng convergence
            verbose: In thông tin debug
            seed: Random seed
        """
        super().__init__(population_size, max_iterations, tolerance, verbose, seed)
        
        self.pa = pa
        self.alpha = alpha
        self.beta = beta
        
        # Problem-specific attributes
        self.objective_func = None
        self.dimensions = None
        self.lower_bound = None
        self.upper_bound = None
        
        # CS-specific attributes
        self.nests = None
        self.fitness_values = None
        self.best_nest = None
        self.best_fitness = float('inf')
    
    def _initialize(self, problem: dict) -> None:
        """
        Khởi tạo CS với problem definition.
        """
        self.objective_func = problem['objective_func']
        self.dimensions = problem['dimensions']
        self.lower_bound = problem.get('lower_bound', -100)
        self.upper_bound = problem.get('upper_bound', 100)
        
        # Initialize nests
        self.nests = self._initialize_random_population(
            self.dimensions,
            self.lower_bound,
            self.upper_bound
        )
        
        # Evaluate
        self.fitness_values = self._evaluate_population(
            self.nests,
            self.objective_func
        )
        
        # Track best
        best_idx = np.argmin(self.fitness_values)
        self.best_nest = self.nests[best_idx].copy()
        self.best_fitness = self.fitness_values[best_idx]
    
    def _iterate(self) -> float:
        """
        Thực hiện một iteration của CS.
        """
        # Generate new solutions via Lévy flights
        new_nests = self._levy_flight()
        
        # Evaluate new solutions
        new_fitness = self._evaluate_population(new_nests, self.objective_func)
        
        # Update if better
        for i in range(self.population_size):
            if new_fitness[i] < self.fitness_values[i]:
                self.nests[i] = new_nests[i]
                self.fitness_values[i] = new_fitness[i]
        
        # Abandon worst nests (pa fraction)
        self._abandon_worst_nests()
        
        # Update best
        best_idx = np.argmin(self.fitness_values)
        if self.fitness_values[best_idx] < self.best_fitness:
            self.best_nest = self.nests[best_idx].copy()
            self.best_fitness = self.fitness_values[best_idx]
        self.position_history.append(self.nests.copy())
        self.best_history.append(self.best_nest.copy())

        return self.best_fitness
    
    def _levy_flight(self) -> np.ndarray:
        """
        Generate new solutions using Lévy flights.
        
        Returns:
            new_nests: New candidate solutions
        """
        # Mantegna's method for Lévy distribution
        sigma_u = (
            np.math.gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2) /
            (np.math.gamma((1 + self.beta) / 2) * self.beta * 2**((self.beta - 1) / 2))
        ) ** (1 / self.beta)
        
        new_nests = np.zeros_like(self.nests)
        
        for i in range(self.population_size):
            # Random walk
            u = np.random.normal(0, sigma_u, size=self.dimensions)
            v = np.random.normal(0, 1, size=self.dimensions)
            
            step = u / (np.abs(v) ** (1 / self.beta))
            
            # Scale
            scale = self.alpha * step * (self.nests[i] - self.best_nest)
            
            # Generate new nest
            new_nests[i] = self.nests[i] + scale
            
            # Boundary handling
            new_nests[i] = np.clip(
                new_nests[i],
                self.lower_bound,
                self.upper_bound
            )
        
        return new_nests
    
    def _abandon_worst_nests(self) -> None:
        """
        Abandon pa fraction of worst nests and generate new ones.
        """
        n_abandon = int(self.pa * self.population_size)
        
        if n_abandon > 0:
            # Find worst nests
            worst_indices = np.argsort(self.fitness_values)[-n_abandon:]
            
            # Generate new random nests
            for i in worst_indices:
                # Generate new position
                self.nests[i] = np.random.uniform(
                    self.lower_bound,
                    self.upper_bound,
                    size=self.dimensions
                )
                
                # Evaluate
                self.fitness_values[i] = self.objective_func(self.nests[i])
                self._evaluations += 1
    
    def _get_best_solution(self) -> Tuple[np.ndarray, float]:
        """Trả về best solution."""
        return self.best_nest.copy(), self.best_fitness
    
    def _get_metadata(self) -> dict:
        """Thêm CS-specific metadata."""
        metadata = super()._get_metadata()
        metadata.update({
            'pa': self.pa,
            'alpha': self.alpha,
            'beta': self.beta
        })
        return metadata