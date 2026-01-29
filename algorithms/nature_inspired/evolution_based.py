"""
Evolution-Based Algorithms
Genetic Algorithm (GA), Differential Evolution (DE)
"""

import numpy as np
from typing import Callable, Tuple, List, Optional
from ..base_optimizer import PopulationBasedOptimizer, OptimizationResults


class GeneticAlgorithm(PopulationBasedOptimizer):
    """
    Genetic Algorithm (GA)
    
    Đặc điểm:
        - Inspired by Darwinian evolution
        - Uses selection, crossover, mutation
        - Population-based stochastic search
    
    Operators:
        - Selection: Tournament, Roulette Wheel, Rank-based
        - Crossover: Single-point, Two-point, Uniform
        - Mutation: Bit-flip, Gaussian
    
    Phù hợp cho:
        - Combinatorial optimization
        - Feature selection
        - Scheduling problems
        - Multi-objective optimization
    """
    
    def __init__(self,
                 population_size: int = 50,
                 max_iterations: int = 1000,
                 bounds: Optional[List[Tuple[float, float]]] = None,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 selection_method: str = 'tournament',
                 crossover_method: str = 'single_point',
                 tournament_size: int = 3,
                 elitism_count: int = 2,
                 tolerance: float = 1e-6,
                 verbose: bool = False,
                 seed: Optional[int] = None):
        """
        Args:
            population_size: Số lượng individuals trong population
            max_iterations: Số generations tối đa
            bounds: Search space bounds - List of (lower, upper) tuples for each dimension
            crossover_rate: Xác suất thực hiện crossover (0.0 - 1.0)
            mutation_rate: Xác suất mutation cho mỗi gene (0.0 - 1.0)
            selection_method: Phương pháp selection ('tournament', 'roulette', 'rank')
            crossover_method: Phương pháp crossover ('single_point', 'two_point', 'uniform')
            tournament_size: Kích thước tournament (cho tournament selection)
            elitism_count: Số best individuals giữ lại mỗi generation
            tolerance: Ngưỡng convergence
            verbose: In thông tin debug
            seed: Random seed
        """
        super().__init__(population_size, max_iterations, tolerance, verbose, seed)
        
        self.bounds = bounds
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.tournament_size = tournament_size
        self.elitism_count = min(elitism_count, population_size)
        
        # Problem-specific attributes (will be set in _initialize)
        self.objective_func = None
        self.dimensions = None
        self.lower_bound = None
        self.upper_bound = None
        self.is_discrete = False
        
        # Population and fitness
        self.fitness_values = None
        self.best_individual = None
        self.best_fitness = float('inf')
    
    def _initialize(self, problem: dict) -> None:
        """
        Khởi tạo GA với problem definition.
        
        Args:
            problem: Dictionary chứa:
                - 'objective_func': Hàm objective cần minimize
                - 'dimensions': Số dimensions
                - 'lower_bound': Giới hạn dưới
                - 'upper_bound': Giới hạn trên
                - 'is_discrete': Boolean - discrete hay continuous problem
        """
        self.objective_func = problem['objective_func']
        self.dimensions = problem['dimensions']
        self.lower_bound = problem.get('lower_bound', -100)
        self.upper_bound = problem.get('upper_bound', 100)
        self.is_discrete = problem.get('is_discrete', False)
        
        # Initialize random population
        if self.is_discrete:
            # For discrete problems (e.g., binary encoding)
            self.population = np.random.randint(
                0, 2, size=(self.population_size, self.dimensions)
            )
        else:
            # For continuous problems
            self.population = self._initialize_random_population(
                self.dimensions,
                self.lower_bound,
                self.upper_bound
            )
        
        # Evaluate initial population
        self.fitness_values = self._evaluate_population(
            self.population,
            self.objective_func
        )
        
        # Track best individual
        best_idx = np.argmin(self.fitness_values)
        self.best_individual = self.population[best_idx].copy()
        self.best_fitness = self.fitness_values[best_idx]
    
    def _iterate(self) -> float:
        """
        Thực hiện một generation của GA.
        
        Returns:
            best_fitness: Best fitness trong generation này
        """
        # Elitism - giữ lại best individuals
        elite_indices = np.argsort(self.fitness_values)[:self.elitism_count]
        elites = self.population[elite_indices].copy()
        
        # Create new population
        new_population = []
        
        # Add elites
        for elite in elites:
            new_population.append(elite)
        
        # Generate offspring
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self._select_parent()
            parent2 = self._select_parent()
            
            # Crossover
            if np.random.random() < self.crossover_rate:
                offspring1, offspring2 = self._crossover(parent1, parent2)
            else:
                offspring1, offspring2 = parent1.copy(), parent2.copy()
            
            # Mutation
            offspring1 = self._mutate(offspring1)
            offspring2 = self._mutate(offspring2)
            
            new_population.append(offspring1)
            if len(new_population) < self.population_size:
                new_population.append(offspring2)
        
        # Update population
        self.population = np.array(new_population[:self.population_size])
        
        # Evaluate new population
        self.fitness_values = self._evaluate_population(
            self.population,
            self.objective_func
        )
        
        # Update best individual
        best_idx = np.argmin(self.fitness_values)
        if self.fitness_values[best_idx] < self.best_fitness:
            self.best_individual = self.population[best_idx].copy()
            self.best_fitness = self.fitness_values[best_idx]
        
        return self.best_fitness
    
    def _select_parent(self) -> np.ndarray:
        """
        Chọn parent dựa trên selection method.
        
        Returns:
            parent: Selected individual
        """
        if self.selection_method == 'tournament':
            return self._tournament_selection()
        elif self.selection_method == 'roulette':
            return self._roulette_wheel_selection()
        elif self.selection_method == 'rank':
            return self._rank_selection()
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
    
    def _tournament_selection(self) -> np.ndarray:
        """Tournament selection - chọn best từ random tournament."""
        tournament_indices = np.random.randint(
            0, self.population_size, size=self.tournament_size
        )
        tournament_fitness = self.fitness_values[tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return self.population[winner_idx].copy()
    
    def _roulette_wheel_selection(self) -> np.ndarray:
        """Roulette wheel selection - xác suất tỉ lệ nghịch với fitness."""
        # Convert to maximization problem
        max_fitness = np.max(self.fitness_values)
        adjusted_fitness = max_fitness - self.fitness_values + 1e-10
        
        # Calculate selection probabilities
        probabilities = adjusted_fitness / np.sum(adjusted_fitness)
        
        # Select based on probabilities
        selected_idx = np.random.choice(
            self.population_size,
            p=probabilities
        )
        return self.population[selected_idx].copy()
    
    def _rank_selection(self) -> np.ndarray:
        """Rank-based selection - xác suất dựa trên rank."""
        ranks = np.argsort(np.argsort(self.fitness_values))
        probabilities = (self.population_size - ranks) / np.sum(range(1, self.population_size + 1))
        
        selected_idx = np.random.choice(
            self.population_size,
            p=probabilities
        )
        return self.population[selected_idx].copy()
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Thực hiện crossover giữa 2 parents.
        
        Returns:
            (offspring1, offspring2): Hai offspring
        """
        if self.crossover_method == 'single_point':
            return self._single_point_crossover(parent1, parent2)
        elif self.crossover_method == 'two_point':
            return self._two_point_crossover(parent1, parent2)
        elif self.crossover_method == 'uniform':
            return self._uniform_crossover(parent1, parent2)
        else:
            raise ValueError(f"Unknown crossover method: {self.crossover_method}")
    
    def _single_point_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Single-point crossover."""
        point = np.random.randint(1, self.dimensions)
        
        offspring1 = np.concatenate([parent1[:point], parent2[point:]])
        offspring2 = np.concatenate([parent2[:point], parent1[point:]])
        
        return offspring1, offspring2
    
    def _two_point_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Two-point crossover."""
        points = sorted(np.random.choice(range(1, self.dimensions), size=2, replace=False))
        
        offspring1 = np.concatenate([
            parent1[:points[0]],
            parent2[points[0]:points[1]],
            parent1[points[1]:]
        ])
        offspring2 = np.concatenate([
            parent2[:points[0]],
            parent1[points[0]:points[1]],
            parent2[points[1]:]
        ])
        
        return offspring1, offspring2
    
    def _uniform_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform crossover - mỗi gene có 50% từ mỗi parent."""
        mask = np.random.randint(0, 2, size=self.dimensions, dtype=bool)
        
        offspring1 = np.where(mask, parent1, parent2)
        offspring2 = np.where(mask, parent2, parent1)
        
        return offspring1, offspring2
    
    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        Thực hiện mutation trên individual.
        
        Returns:
            mutated_individual
        """
        individual = individual.copy()
        
        if self.is_discrete:
            # Bit-flip mutation for binary encoding
            mutation_mask = np.random.random(self.dimensions) < self.mutation_rate
            individual[mutation_mask] = 1 - individual[mutation_mask]
        else:
            # Gaussian mutation for continuous problems
            mutation_mask = np.random.random(self.dimensions) < self.mutation_rate
            if np.any(mutation_mask):
                mutation_range = self.upper_bound - self.lower_bound
                mutations = np.random.normal(0, 0.1 * mutation_range, size=self.dimensions)
                individual[mutation_mask] += mutations[mutation_mask]
                
                # Clip to bounds
                individual = np.clip(individual, self.lower_bound, self.upper_bound)
        
        return individual
    
    def _get_best_solution(self) -> Tuple[np.ndarray, float]:
        """Trả về best solution."""
        return self.best_individual.copy(), self.best_fitness
    
    def _get_metadata(self) -> dict:
        """Thêm GA-specific metadata."""
        metadata = super()._get_metadata()
        metadata.update({
            'crossover_rate': self.crossover_rate,
            'mutation_rate': self.mutation_rate,
            'selection_method': self.selection_method,
            'crossover_method': self.crossover_method,
            'elitism_count': self.elitism_count
        })
        return metadata


class DifferentialEvolution(PopulationBasedOptimizer):
    """
    Differential Evolution (DE)
    
    Đặc điểm:
        - Uses vector differences for mutation
        - Simple and effective for continuous optimization
        - Few control parameters
    
    Strategy: DE/rand/1/bin (default)
        - Mutation: v = x_r1 + F * (x_r2 - x_r3)
        - Crossover: Binomial
    
    Phù hợp cho:
        - Continuous optimization
        - Global optimization
        - Engineering design problems
        - Function optimization
    """
    
    def __init__(self,
                 population_size: int = 50,
                 max_iterations: int = 1000,
                 bounds: Optional[List[Tuple[float, float]]] = None,
                 mutation_factor: float = 0.8,
                 crossover_rate: float = 0.9,
                 strategy: str = 'rand/1/bin',
                 tolerance: float = 1e-6,
                 verbose: bool = False,
                 seed: Optional[int] = None):
        """
        Args:
            population_size: Số lượng individuals
            max_iterations: Số iterations tối đa
            bounds: Search space bounds - List of (lower, upper) tuples for each dimension
            mutation_factor (F): Scale factor cho mutation (0.0 - 2.0)
            crossover_rate (CR): Xác suất crossover (0.0 - 1.0)
            strategy: DE strategy ('rand/1/bin', 'best/1/bin', 'rand/2/bin')
            tolerance: Ngưỡng convergence
            verbose: In thông tin debug
            seed: Random seed
        """
        super().__init__(population_size, max_iterations, tolerance, verbose, seed)
        
        self.bounds = bounds
        self.mutation_factor = mutation_factor
        self.crossover_rate = crossover_rate
        self.strategy = strategy
        
        # Problem-specific attributes
        self.objective_func = None
        self.dimensions = None
        self.lower_bound = None
        self.upper_bound = None
        
        # Population and fitness
        self.fitness_values = None
        self.best_individual = None
        self.best_fitness = float('inf')
    
    def _initialize(self, problem: dict) -> None:
        """
        Khởi tạo DE với problem definition.
        
        Args:
            problem: Dictionary chứa:
                - 'objective_func': Hàm objective cần minimize
                - 'dimensions': Số dimensions
                - 'lower_bound': Giới hạn dưới
                - 'upper_bound': Giới hạn trên
        """
        self.objective_func = problem['objective_func']
        self.dimensions = problem['dimensions']
        self.lower_bound = problem.get('lower_bound', -100)
        self.upper_bound = problem.get('upper_bound', 100)
        
        # Initialize random population
        self.population = self._initialize_random_population(
            self.dimensions,
            self.lower_bound,
            self.upper_bound
        )
        
        # Evaluate initial population
        self.fitness_values = self._evaluate_population(
            self.population,
            self.objective_func
        )
        
        # Track best individual
        best_idx = np.argmin(self.fitness_values)
        self.best_individual = self.population[best_idx].copy()
        self.best_fitness = self.fitness_values[best_idx]
    
    def _iterate(self) -> float:
        """
        Thực hiện một iteration của DE.
        
        Returns:
            best_fitness: Best fitness trong iteration này
        """
        new_population = []
        new_fitness = []
        
        for i in range(self.population_size):
            # Mutation
            mutant = self._mutate(i)
            
            # Crossover
            trial = self._crossover(self.population[i], mutant)
            
            # Boundary handling
            trial = np.clip(trial, self.lower_bound, self.upper_bound)
            
            # Evaluation
            trial_fitness = self.objective_func(trial)
            self._evaluations += 1
            
            # Selection
            if trial_fitness < self.fitness_values[i]:
                new_population.append(trial)
                new_fitness.append(trial_fitness)
                
                # Update global best
                if trial_fitness < self.best_fitness:
                    self.best_individual = trial.copy()
                    self.best_fitness = trial_fitness
            else:
                new_population.append(self.population[i])
                new_fitness.append(self.fitness_values[i])
        
        # Update population
        self.population = np.array(new_population)
        self.fitness_values = np.array(new_fitness)
        
        return self.best_fitness
    
    def _mutate(self, target_idx: int) -> np.ndarray:
        """
        Tạo mutant vector dựa trên strategy.
        
        Args:
            target_idx: Index của target vector
            
        Returns:
            mutant: Mutant vector
        """
        if self.strategy == 'rand/1/bin':
            return self._mutate_rand_1(target_idx)
        elif self.strategy == 'best/1/bin':
            return self._mutate_best_1(target_idx)
        elif self.strategy == 'rand/2/bin':
            return self._mutate_rand_2(target_idx)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _mutate_rand_1(self, target_idx: int) -> np.ndarray:
        """
        DE/rand/1 mutation: v = x_r1 + F * (x_r2 - x_r3)
        """
        # Select 3 random distinct individuals (different from target)
        candidates = [i for i in range(self.population_size) if i != target_idx]
        r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)
        
        # Mutation
        mutant = self.population[r1] + self.mutation_factor * (
            self.population[r2] - self.population[r3]
        )
        
        return mutant
    
    def _mutate_best_1(self, target_idx: int) -> np.ndarray:
        """
        DE/best/1 mutation: v = x_best + F * (x_r1 - x_r2)
        """
        # Select 2 random distinct individuals (different from target)
        candidates = [i for i in range(self.population_size) if i != target_idx]
        r1, r2 = np.random.choice(candidates, size=2, replace=False)
        
        # Mutation
        best_idx = np.argmin(self.fitness_values)
        mutant = self.population[best_idx] + self.mutation_factor * (
            self.population[r1] - self.population[r2]
        )
        
        return mutant
    
    def _mutate_rand_2(self, target_idx: int) -> np.ndarray:
        """
        DE/rand/2 mutation: v = x_r1 + F * (x_r2 - x_r3) + F * (x_r4 - x_r5)
        """
        # Select 5 random distinct individuals (different from target)
        candidates = [i for i in range(self.population_size) if i != target_idx]
        r1, r2, r3, r4, r5 = np.random.choice(candidates, size=5, replace=False)
        
        # Mutation
        mutant = self.population[r1] + self.mutation_factor * (
            self.population[r2] - self.population[r3]
        ) + self.mutation_factor * (
            self.population[r4] - self.population[r5]
        )
        
        return mutant
    
    def _crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """
        Binomial crossover giữa target và mutant.
        
        Args:
            target: Target vector
            mutant: Mutant vector
            
        Returns:
            trial: Trial vector
        """
        trial = target.copy()
        
        # Ensure at least one dimension from mutant
        j_rand = np.random.randint(0, self.dimensions)
        
        for j in range(self.dimensions):
            if np.random.random() < self.crossover_rate or j == j_rand:
                trial[j] = mutant[j]
        
        return trial
    
    def _get_best_solution(self) -> Tuple[np.ndarray, float]:
        """Trả về best solution."""
        return self.best_individual.copy(), self.best_fitness
    
    def _get_metadata(self) -> dict:
        """Thêm DE-specific metadata."""
        metadata = super()._get_metadata()
        metadata.update({
            'mutation_factor': self.mutation_factor,
            'crossover_rate': self.crossover_rate,
            'strategy': self.strategy
        })
        return metadata