import time
import numpy as np

class ExperimentLogger:
    def __init__(self):
        self.history = []        # Lưu fitness qua từng iteration (cho Sphere, Rastrigin, TSP)
        self.best_fitness = float('inf')
        self.best_solution = None
        self.nodes_explored = 0  # Dành riêng cho Grid Map (BFS, DFS, A*)
        self.start_time = 0
        self.execution_time = 0

    def start_timer(self):
        self.start_time = time.perf_counter()

    def stop_timer(self):
        self.execution_time = time.perf_counter() - self.start_time

    def log_optimization(self, iteration, fitness, solution=None):
        """Dùng cho Sphere, Rastrigin, TSP"""
        self.history.append(fitness)
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            if solution is not None:
                self.best_solution = np.array(solution)

    def log_search(self, nodes_count, path_cost, path):
        """Dùng cho Grid Map"""
        self.nodes_explored = nodes_count
        self.best_fitness = path_cost # Trong graph search, fitness là cost
        self.best_solution = path