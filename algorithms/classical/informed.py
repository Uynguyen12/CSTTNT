"""
Informed Search Algorithms
Greedy Best-First Search, A* - sử dụng heuristic
"""

import heapq
import numpy as np
from typing import Tuple

from .base_search import InformedSearchAlgorithm, SearchResults


class GreedyBestFirstSearch(InformedSearchAlgorithm):
    """
    Greedy Best-First Search
    
    Đặc điểm:
        - Complete: No (có thể bị cuốn vào wrong path)
        - Optimal: No (tìm được path nhưng không phải optimal)
        - Time Complexity: O(b^m)
        - Space Complexity: O(b^m)
    
    Sử dụng priority queue ordered by heuristic function h(n).
    Chỉ quan tâm đến estimated distance tới goal, không quan tâm cost từ start.
    """
    
    def __init__(self, heuristic='euclidean', verbose: bool = False):
        super().__init__(heuristic=heuristic, verbose=verbose)
    
    def _search(self, graph, start: int, goal: int) -> SearchResults:
        """Greedy Best-First Search algorithm implementation."""
        # Initialize
        # Priority queue: (h_value, node)
        pq = [(0.0, start)]
        
        visited = np.zeros(graph.num_nodes, dtype=bool)
        parent = np.full(graph.num_nodes, -1, dtype=int)
        
        # Greedy loop
        while pq:
            _, current = heapq.heappop(pq)
            
            # Skip if already visited
            if visited[current]:
                continue
            
            visited[current] = True
            self._explored_order.append(current)
            self._nodes_explored += 1
            
            # Goal test
            if current == goal:
                path = self._reconstruct_path(parent, start, goal)
                # Calculate actual path cost
                cost = self._calculate_path_cost(graph, path)
                return SearchResults(
                    path=path,
                    cost=cost,
                    path_found=True,
                    nodes_explored=self._nodes_explored,
                    explored_order=self._explored_order.copy()
                )
            
            # Expand neighbors
            for neighbor, _ in graph.get_neighbors(current):
                if not visited[neighbor]:
                    parent[neighbor] = current
                    
                    # Calculate heuristic value
                    h_value = self._compute_heuristic(graph, neighbor, goal)
                    heapq.heappush(pq, (h_value, neighbor))
        
        # No path found
        return SearchResults(
            path_found=False,
            nodes_explored=self._nodes_explored,
            explored_order=self._explored_order.copy()
        )
    
    def _calculate_path_cost(self, graph, path):
        """Calculate actual cost of a path."""
        if len(path) <= 1:
            return 0.0
        
        total_cost = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            neighbors = dict(graph.get_neighbors(u))
            if v in neighbors:
                total_cost += neighbors[v]
        
        return total_cost


class AStarSearch(InformedSearchAlgorithm):
    """
    A* Search Algorithm
    
    Đặc điểm:
        - Complete: Yes
        - Optimal: Yes (với admissible heuristic)
        - Time Complexity: O(b^d)
        - Space Complexity: O(b^d)
    
    Sử dụng priority queue ordered by f(n) = g(n) + h(n).
    g(n) = actual cost từ start đến node n
    h(n) = heuristic estimate từ node n tới goal
    f(n) = total estimated cost
    
    Là best informed search algorithm vì combine:
    - Actual cost đã đi (g)
    - Estimate còn phải đi (h)
    """
    
    def __init__(self, heuristic='euclidean', verbose: bool = False):
        super().__init__(heuristic=heuristic, verbose=verbose)
    
    def _search(self, graph, start: int, goal: int) -> SearchResults:
        """A* Search algorithm implementation."""
        # Initialize
        # Priority queue: (f_value, counter, node)
        # Counter để tránh comparison giữa nodes
        counter = 0
        pq = [(0.0, counter, start)]
        counter += 1
        
        # Track best g value (actual cost) to each node
        g_score = np.full(graph.num_nodes, np.inf)
        g_score[start] = 0.0
        
        parent = np.full(graph.num_nodes, -1, dtype=int)
        visited = np.zeros(graph.num_nodes, dtype=bool)
        
        # A* loop
        while pq:
            f_value, _, current = heapq.heappop(pq)
            
            # Skip if already visited
            if visited[current]:
                continue
            
            visited[current] = True
            self._explored_order.append(current)
            self._nodes_explored += 1
            
            # Goal test
            if current == goal:
                path = self._reconstruct_path(parent, start, goal)
                return SearchResults(
                    path=path,
                    cost=g_score[goal],
                    path_found=True,
                    nodes_explored=self._nodes_explored,
                    explored_order=self._explored_order.copy()
                )
            
            # Expand neighbors
            for neighbor, edge_cost in graph.get_neighbors(current):
                new_g = g_score[current] + edge_cost
                
                # Update nếu tìm được đường đi tốt hơn
                if new_g < g_score[neighbor]:
                    parent[neighbor] = current
                    g_score[neighbor] = new_g
                    
                    # Calculate h and f values
                    h_value = self._compute_heuristic(graph, neighbor, goal)
                    f_value = new_g + h_value
                    
                    heapq.heappush(pq, (f_value, counter, neighbor))
                    counter += 1
        
        # No path found
        return SearchResults(
            path_found=False,
            nodes_explored=self._nodes_explored,
            explored_order=self._explored_order.copy()
        )


# Factory function để tạo algorithms dễ dàng
def create_informed_search(algorithm: str, 
                           heuristic: str = 'euclidean',
                           verbose: bool = False):
    """
    Factory function để tạo informed search algorithms.
    
    Args:
        algorithm: Tên thuật toán ('greedy', 'astar')
        heuristic: Loại heuristic ('euclidean' hoặc 'manhattan')
        verbose: In thông tin debug
        
    Returns:
        Instance của thuật toán tương ứng
        
    Example:
        >>> astar = create_informed_search('astar', heuristic='euclidean')
        >>> result = astar.search(graph, start, goal)
    """
    algorithms = {
        'greedy': GreedyBestFirstSearch,
        'astar': AStarSearch
    }
    
    if algorithm.lower() not in algorithms:
        raise ValueError(f"Unknown algorithm: {algorithm}. "
                        f"Choose from {list(algorithms.keys())}")
    
    return algorithms[algorithm.lower()](
        heuristic=heuristic,
        verbose=verbose
    )
