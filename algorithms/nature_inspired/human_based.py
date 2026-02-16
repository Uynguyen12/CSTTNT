"""
Human Behavior-Based Algorithms
Teaching-Learning-Based Optimization (TLBO)
"""

import numpy as np
from typing import Callable, Tuple, List, Optional
from ..base_optimizer import PopulationBasedOptimizer, OptimizationResults


class TeachingLearningBasedOptimization(PopulationBasedOptimizer):
    """
    Teaching-Learning-Based Optimization (TLBO)
    
    Đặc điểm:
        - Inspired by teaching-learning process in classroom
        - Two phases: Teacher phase and Learner phase
        - Parameter-free (no algorithm-specific parameters)
        - Simple and effective
    
    Teacher Phase:
        - Best student becomes teacher
        - Students learn from teacher
    
    Learner Phase:
        - Students learn from each other
        - Random interaction
    
    Phù hợp cho:
        - Continuous optimization
        - Engineering design
        - Parameter tuning
        - Constrained optimization
    """
    
    def __init__(self,
                 population_size: int = 50,
                 max_iterations: int = 1000,
                 tolerance: float = 1e-6,
                 verbose: bool = False,
                 seed: Optional[int] = None):
        """
        Args:
            population_size: Số lượng students (learners)
            max_iterations: Số iterations tối đa
            tolerance: Ngưỡng convergence
            verbose: In thông tin debug
            seed: Random seed
            
        Note: TLBO is parameter-free - không có algorithm-specific parameters
        """
        super().__init__(population_size, max_iterations, tolerance, verbose, seed)
        
        # Problem-specific attributes
        self.objective_func = None
        self.dimensions = None
        self.lower_bound = None
        self.upper_bound = None
        
        # TLBO-specific attributes
        self.students = None  # Population (learners)
        self.fitness_values = None
        self.teacher = None  # Best student
        self.teacher_fitness = float('inf')
        self.best_solution = None
        self.best_fitness = float('inf')
    
    def _initialize(self, problem: dict) -> None:
        """
        Khởi tạo TLBO với problem definition.
        
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
        
        # Initialize students (random population)
        self.students = self._initialize_random_population(
            self.dimensions,
            self.lower_bound,
            self.upper_bound
        )
        
        # Evaluate initial population
        self.fitness_values = self._evaluate_population(
            self.students,
            self.objective_func
        )
        
        # Find teacher (best student)
        teacher_idx = np.argmin(self.fitness_values)
        self.teacher = self.students[teacher_idx].copy()
        self.teacher_fitness = self.fitness_values[teacher_idx]
        
        # Track best
        self.best_solution = self.teacher.copy()
        self.best_fitness = self.teacher_fitness
    
    def _iterate(self) -> float:
        """
        Thực hiện một iteration của TLBO.
        Bao gồm Teacher Phase và Learner Phase.
        
        Returns:
            best_fitness: Best fitness trong iteration này
        """
        # Teacher Phase
        self._teacher_phase()
        
        # Learner Phase
        self._learner_phase()
        
        # Update teacher (best student)
        teacher_idx = np.argmin(self.fitness_values)
        if self.fitness_values[teacher_idx] < self.teacher_fitness:
            self.teacher = self.students[teacher_idx].copy()
            self.teacher_fitness = self.fitness_values[teacher_idx]
            
            # Update global best
            if self.teacher_fitness < self.best_fitness:
                self.best_solution = self.teacher.copy()
                self.best_fitness = self.teacher_fitness
                
        self.position_history.append(self.students.copy())
        self.best_history.append(self.best_solution.copy())

        return self.best_fitness
    
    def _teacher_phase(self) -> None:
        """
        Teacher Phase: Students learn from teacher.
        
        Equation:
            X_new = X_old + r * (Teacher - T_F * Mean)
        where:
            T_F = Teaching Factor (1 or 2, random)
            Mean = Mean of all students
            r = random number [0, 1]
        """
        # Calculate mean of all students
        mean = np.mean(self.students, axis=0)
        
        # For each student
        for i in range(self.population_size):
            # Teaching factor (randomly 1 or 2)
            T_F = np.random.choice([1, 2])
            
            # Random factor
            r = np.random.random(self.dimensions)
            
            # New position
            new_student = self.students[i] + r * (
                self.teacher - T_F * mean
            )
            
            # Boundary handling
            new_student = np.clip(
                new_student,
                self.lower_bound,
                self.upper_bound
            )
            
            # Evaluate
            new_fitness = self.objective_func(new_student)
            self._evaluations += 1
            
            # Accept if better
            if new_fitness < self.fitness_values[i]:
                self.students[i] = new_student
                self.fitness_values[i] = new_fitness
    
    def _learner_phase(self) -> None:
        """
        Learner Phase: Students learn from each other.
        
        Equation:
            If f(X_i) < f(X_j):
                X_new = X_i + r * (X_i - X_j)
            Else:
                X_new = X_i + r * (X_j - X_i)
        where:
            X_j = randomly selected student (j != i)
            r = random number [0, 1]
        """
        # For each student
        for i in range(self.population_size):
            # Select another random student
            j = i
            while j == i:
                j = np.random.randint(0, self.population_size)
            
            # Random factor
            r = np.random.random(self.dimensions)
            
            # Learn from better student
            if self.fitness_values[i] < self.fitness_values[j]:
                new_student = self.students[i] + r * (
                    self.students[i] - self.students[j]
                )
            else:
                new_student = self.students[i] + r * (
                    self.students[j] - self.students[i]
                )
            
            # Boundary handling
            new_student = np.clip(
                new_student,
                self.lower_bound,
                self.upper_bound
            )
            
            # Evaluate
            new_fitness = self.objective_func(new_student)
            self._evaluations += 1
            
            # Accept if better
            if new_fitness < self.fitness_values[i]:
                self.students[i] = new_student
                self.fitness_values[i] = new_fitness
    
    def _get_best_solution(self) -> Tuple[np.ndarray, float]:
        """
        Trả về best solution.
        
        Returns:
            (solution, fitness): Best solution và fitness của nó
        """
        return self.best_solution.copy(), self.best_fitness
    
    def _get_metadata(self) -> dict:
        """
        Thêm TLBO-specific metadata.
        
        Returns:
            metadata: Dictionary chứa thông tin về algorithm
        """
        metadata = super()._get_metadata()
        metadata.update({
            'parameter_free': True,
            'note': 'TLBO requires no algorithm-specific parameters'
        })
        return metadata