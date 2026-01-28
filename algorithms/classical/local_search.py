"""
Local Search Algorithms
Hill Climbing (Steepest Ascent), Simulated Annealing
"""

import numpy as np
import random
import math
from typing import Tuple

from .base_search import LocalSearchAlgorithm, SearchResults


class HillClimbingSearch(LocalSearchAlgorithm):
    """
    Hill Climbing (Steepest Ascent)
    
    Đặc điểm:
        - Complete: No (bị mắc kẹt ở local maxima)
        - Optimal: No
        - Time Complexity: O(number_of_neighbors)
        - Space Complexity: O(1)
    
    Chiến lược:
        - Bắt đầu từ random node
        - Luôn di chuyển đến neighbor có heuristic value tốt nhất
        - Dừng lại khi không tìm được neighbor tốt hơn
    
    Phù hợp cho local optimization không cần global optimal.
    """
    
    def __init__(self, heuristic: str = 'euclidean', verbose: bool = False):
        super().__init__(heuristic=heuristic, verbose=verbose)
    
    def _search(self, graph, start: int, goal: int) -> SearchResults:
        """Hill Climbing (Steepest Ascent) algorithm implementation."""
        
        current = start
        self._explored_order.append(current)
        self._nodes_explored += 1
        
        current_h = self._compute_heuristic(graph, current, goal)
        
        # Hill climbing loop - di chuyển đến neighbor tốt nhất
        improved = True
        while improved:
            improved = False
            best_neighbor = None
            best_h = current_h
            
            # Kiểm tra tất cả neighbors
            neighbors = graph.get_neighbors(current)
            for neighbor, _ in neighbors:
                # Tránh revisit nodes trong local search
                if neighbor not in self._explored_order or neighbor == goal:
                    neighbor_h = self._compute_heuristic(graph, neighbor, goal)
                    
                    # Chỉ move nếu neighbor tốt hơn (steepest ascent)
                    if neighbor_h < best_h:
                        best_neighbor = neighbor
                        best_h = neighbor_h
                        improved = True
            
            # Move to best neighbor
            if improved and best_neighbor is not None:
                current = best_neighbor
                current_h = best_h
                self._explored_order.append(current)
                self._nodes_explored += 1
                
                # Goal test
                if current == goal:
                    # Reconstruct path từ explored_order
                    path = self._reconstruct_hill_climbing_path(graph, start, current)
                    cost = self._calculate_path_cost(graph, path)
                    return SearchResults(
                        path=path,
                        cost=cost,
                        path_found=True,
                        nodes_explored=self._nodes_explored,
                        explored_order=self._explored_order.copy()
                    )
        
        # Không tìm được goal
        path = self._reconstruct_hill_climbing_path(graph, start, current)
        cost = self._calculate_path_cost(graph, path)
        
        if current == goal:
            return SearchResults(
                path=path,
                cost=cost,
                path_found=True,
                nodes_explored=self._nodes_explored,
                explored_order=self._explored_order.copy()
            )
        else:
            return SearchResults(
                path=path,
                cost=cost,
                path_found=False,
                nodes_explored=self._nodes_explored,
                explored_order=self._explored_order.copy()
            )
    
    def _reconstruct_hill_climbing_path(self, graph, start: int, current: int):
        """
        Rebuild path từ start đến current dựa trên explored_order.
        Đây là greedy path reconstruction.
        """
        # Simple greedy path từ start
        path = [start]
        current_node = start
        
        # Try to find neighbors in explored_order to reconstruct path
        visited = set()
        visited.add(current_node)
        
        while current_node != current:
            neighbors = graph.get_neighbors(current_node)
            next_node = None
            best_h = float('inf')
            
            for neighbor, _ in neighbors:
                if neighbor not in visited:
                    h = self._compute_heuristic(graph, neighbor, current)
                    if h < best_h:
                        best_h = h
                        next_node = neighbor
            
            if next_node is None:
                break
            
            path.append(next_node)
            visited.add(next_node)
            current_node = next_node
        
        return path
    
    def _calculate_path_cost(self, graph, path):
        """Calculate actual cost of a path."""
        if len(path) <= 1:
            return 0.0
        
        total_cost = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            neighbors_dict = dict(graph.get_neighbors(u))
            if v in neighbors_dict:
                total_cost += neighbors_dict[v]
        
        return total_cost


class SimulatedAnnealingSearch(LocalSearchAlgorithm):
    """
    Simulated Annealing Search
    
    Đặc điểm:
        - Complete: No (nhưng higher chance thoát khỏi local minima)
        - Optimal: No
        - Time Complexity: Phụ thuộc vào số iterations
        - Space Complexity: O(1)
    
    Chiến lược:
        - Bắt đầu từ start node
        - Random di chuyển đến neighbors
        - Accept moves xấu với xác suất theo temperature
        - Temperature giảm dần theo iterations
        - Xác suất accept moves xấu giảm theo thời gian
    
    Phương trình acceptance:
        P(accept) = exp((f_current - f_neighbor) / temperature)
    
    Phù hợp cho escaping local optima trong optimization problems.
    """
    
    def __init__(self,
                 heuristic: str = 'euclidean',
                 max_iterations: int = 1000,
                 initial_temperature: float = 100.0,
                 cooling_rate: float = 0.95,
                 verbose: bool = False):
        """
        Args:
            heuristic: Loại heuristic ('euclidean' hoặc 'manhattan')
            max_iterations: Số iterations tối đa
            initial_temperature: Nhiệt độ ban đầu (cao -> accept xấu easily)
            cooling_rate: Tỷ lệ giảm nhiệt độ (0.95 = 5% giảm mỗi iteration)
            verbose: In thông tin debug
        """
        super().__init__(heuristic=heuristic, verbose=verbose)
        self.max_iterations = max_iterations
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
    
    def _search(self, graph, start: int, goal: int) -> SearchResults:
        """Simulated Annealing algorithm implementation."""
        
        current = start
        current_h = self._compute_heuristic(graph, current, goal)
        
        best = current
        best_h = current_h
        
        temperature = self.initial_temperature
        
        self._explored_order.append(current)
        self._nodes_explored += 1
        
        # Simulated Annealing loop
        for iteration in range(self.max_iterations):
            # Goal test
            if current == goal:
                path = self._reconstruct_simulated_annealing_path(graph, start, current)
                cost = self._calculate_path_cost(graph, path)
                return SearchResults(
                    path=path,
                    cost=cost,
                    path_found=True,
                    nodes_explored=self._nodes_explored,
                    explored_order=self._explored_order.copy(),
                    metadata={'iterations': iteration}
                )
            
            # Random select neighbor
            neighbors = graph.get_neighbors(current)
            if not neighbors:
                break
            
            neighbor, _ = random.choice(neighbors)
            neighbor_h = self._compute_heuristic(graph, neighbor, goal)
            
            # Calculate acceptance probability
            delta_h = neighbor_h - current_h
            if delta_h < 0 or random.random() < self._acceptance_probability(delta_h, temperature):
                # Accept move
                current = neighbor
                current_h = neighbor_h
                
                if neighbor not in self._explored_order:
                    self._explored_order.append(neighbor)
                    self._nodes_explored += 1
                
                # Track best solution
                if current_h < best_h:
                    best = current
                    best_h = current_h
            
            # Cool down
            temperature *= self.cooling_rate
            
            # Early stopping nếu temperature quá thấp
            if temperature < 1e-8:
                break
        
        # Return best solution found
        path = self._reconstruct_simulated_annealing_path(graph, start, best)
        cost = self._calculate_path_cost(graph, path)
        
        if best == goal:
            return SearchResults(
                path=path,
                cost=cost,
                path_found=True,
                nodes_explored=self._nodes_explored,
                explored_order=self._explored_order.copy(),
                metadata={'iterations': self.max_iterations}
            )
        else:
            return SearchResults(
                path=path,
                cost=cost,
                path_found=False,
                nodes_explored=self._nodes_explored,
                explored_order=self._explored_order.copy(),
                metadata={'iterations': self.max_iterations}
            )
    
    def _acceptance_probability(self, delta_h: float, temperature: float) -> float:
        """
        Calculate acceptance probability của move xấu.
        
        Args:
            delta_h: Sự thay đổi heuristic value (positive = move xấu)
            temperature: Nhiệt độ hiện tại
            
        Returns:
            Xác suất accept move
        """
        if temperature < 1e-8:
            return 0.0
        
        return math.exp(-delta_h / temperature)
    
    def _reconstruct_simulated_annealing_path(self, graph, start: int, current: int):
        """
        Rebuild path từ start đến current.
        Dùng greedy heuristic-based path reconstruction.
        """
        path = [start]
        current_node = start
        visited = set()
        visited.add(current_node)
        
        max_steps = graph.num_nodes * 2  # Avoid infinite loop
        steps = 0
        
        while current_node != current and steps < max_steps:
            neighbors = graph.get_neighbors(current_node)
            next_node = None
            best_h = float('inf')
            
            # Choose neighbor closest to target
            for neighbor, _ in neighbors:
                if neighbor not in visited or neighbor == current:
                    h = self._compute_heuristic(graph, neighbor, current)
                    if h < best_h:
                        best_h = h
                        next_node = neighbor
            
            if next_node is None:
                break
            
            path.append(next_node)
            visited.add(next_node)
            current_node = next_node
            steps += 1
        
        return path
    
    def _calculate_path_cost(self, graph, path):
        """Calculate actual cost of a path."""
        if len(path) <= 1:
            return 0.0
        
        total_cost = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            neighbors_dict = dict(graph.get_neighbors(u))
            if v in neighbors_dict:
                total_cost += neighbors_dict[v]
        
        return total_cost


# Factory function để tạo algorithms dễ dàng
def create_local_search(algorithm: str,
                        heuristic: str = 'euclidean',
                        max_iterations: int = 1000,
                        temperature: float = 100.0,
                        cooling_rate: float = 0.95,
                        verbose: bool = False):
    """
    Factory function để tạo local search algorithms.
    
    Args:
        algorithm: Tên thuật toán ('hillclimbing', 'simulated_annealing')
        heuristic: Loại heuristic ('euclidean' hoặc 'manhattan')
        max_iterations: Số iterations tối đa (cho SA)
        temperature: Nhiệt độ ban đầu (cho SA)
        cooling_rate: Tỷ lệ giảm nhiệt độ (cho SA)
        verbose: In thông tin debug
        
    Returns:
        Instance của thuật toán tương ứng
        
    Example:
        >>> hc = create_local_search('hillclimbing')
        >>> result = hc.search(graph, start, goal)
        
        >>> sa = create_local_search('simulated_annealing', 
        ...                           max_iterations=5000,
        ...                           temperature=50.0)
        >>> result = sa.search(graph, start, goal)
    """
    algorithms = {
        'hillclimbing': HillClimbingSearch,
        'simulated_annealing': SimulatedAnnealingSearch
    }
    
    if algorithm.lower() not in algorithms:
        raise ValueError(f"Unknown algorithm: {algorithm}. "
                        f"Choose from {list(algorithms.keys())}")
    
    if algorithm.lower() == 'hillclimbing':
        return HillClimbingSearch(heuristic=heuristic, verbose=verbose)
    else:
        return SimulatedAnnealingSearch(
            heuristic=heuristic,
            max_iterations=max_iterations,
            initial_temperature=temperature,
            cooling_rate=cooling_rate,
            verbose=verbose
        )
