"""
Uninformed Search Algorithms
BFS, DFS, UCS - không sử dụng heuristic
"""

from collections import deque
import heapq
import numpy as np
from typing import Tuple

from .base_search import UninformedSearchAlgorithm, SearchResults


class BreadthFirstSearch(UninformedSearchAlgorithm):
    """
    Breadth-First Search (BFS)
    
    Đặc điểm:
        - Complete: Yes
        - Optimal: Yes (với uniform cost)
        - Time Complexity: O(b^d)
        - Space Complexity: O(b^d)
    
    Sử dụng queue (FIFO) để explore nodes theo breadth-first manner.
    """
    
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
    
    def _search(self, graph, start: int, goal: int) -> SearchResults:
        """BFS algorithm implementation."""
        # Initialize
        queue = deque([start])
        visited = np.zeros(graph.num_nodes, dtype=bool)
        visited[start] = True
        parent = np.full(graph.num_nodes, -1, dtype=int)
        
        # BFS loop
        while queue:
            current = queue.popleft()
            self._explored_order.append(current)
            self._nodes_explored += 1
            
            # Goal test
            if current == goal:
                path = self._reconstruct_path(parent, start, goal)
                return SearchResults(
                    path=path,
                    cost=len(path) - 1,  # BFS counts steps, not edge weights
                    path_found=True,
                    nodes_explored=self._nodes_explored,
                    explored_order=self._explored_order.copy()
                )
            
            # Expand neighbors
            for neighbor, _ in graph.get_neighbors(current):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    parent[neighbor] = current
                    queue.append(neighbor)
        
        # No path found
        return SearchResults(
            path_found=False,
            nodes_explored=self._nodes_explored,
            explored_order=self._explored_order.copy()
        )


class DepthFirstSearch(UninformedSearchAlgorithm):
    """
    Depth-First Search (DFS)
    
    Đặc điểm:
        - Complete: No (có thể bị loop infinitely)
        - Optimal: No
        - Time Complexity: O(b^m) với m = maximum depth
        - Space Complexity: O(bm)
    
    Sử dụng stack (LIFO) để explore nodes theo depth-first manner.
    """
    
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
    
    def _search(self, graph, start: int, goal: int) -> SearchResults:
        """DFS algorithm implementation."""
        # Initialize
        stack = [start]
        visited = np.zeros(graph.num_nodes, dtype=bool)
        visited[start] = True
        parent = np.full(graph.num_nodes, -1, dtype=int)
        
        # DFS loop
        while stack:
            current = stack.pop()
            self._explored_order.append(current)
            self._nodes_explored += 1
            
            # Goal test
            if current == goal:
                path = self._reconstruct_path(parent, start, goal)
                return SearchResults(
                    path=path,
                    cost=len(path) - 1,
                    path_found=True,
                    nodes_explored=self._nodes_explored,
                    explored_order=self._explored_order.copy()
                )
            
            # Expand neighbors (reversed để maintain consistent order)
            neighbors = graph.get_neighbors(current)
            for neighbor, _ in reversed(neighbors):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    parent[neighbor] = current
                    stack.append(neighbor)
        
        # No path found
        return SearchResults(
            path_found=False,
            nodes_explored=self._nodes_explored,
            explored_order=self._explored_order.copy()
        )


class UniformCostSearch(UninformedSearchAlgorithm):
    """
    Uniform Cost Search (UCS)
    
    Đặc điểm:
        - Complete: Yes
        - Optimal: Yes (tìm đường đi có cost thấp nhất)
        - Time Complexity: O(b^(1 + C*/ε))
        - Space Complexity: O(b^(1 + C*/ε))
    
    Sử dụng priority queue (min-heap) ordered by path cost.
    """
    
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
    
    def _search(self, graph, start: int, goal: int) -> SearchResults:
        """UCS algorithm implementation."""
        # Initialize
        # Priority queue: (cost, node)
        pq = [(0.0, start)]
        
        # Track best cost to reach each node
        cost_so_far = np.full(graph.num_nodes, np.inf)
        cost_so_far[start] = 0.0
        
        parent = np.full(graph.num_nodes, -1, dtype=int)
        visited = np.zeros(graph.num_nodes, dtype=bool)
        
        # UCS loop
        while pq:
            current_cost, current = heapq.heappop(pq)
            
            # Skip if already visited với lower cost
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
                    cost=current_cost,
                    path_found=True,
                    nodes_explored=self._nodes_explored,
                    explored_order=self._explored_order.copy()
                )
            
            # Expand neighbors
            for neighbor, edge_cost in graph.get_neighbors(current):
                new_cost = current_cost + edge_cost
                
                # Update nếu tìm được đường đi tốt hơn
                if new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    parent[neighbor] = current
                    heapq.heappush(pq, (new_cost, neighbor))
        
        # No path found
        return SearchResults(
            path_found=False,
            nodes_explored=self._nodes_explored,
            explored_order=self._explored_order.copy()
        )


# Factory function để tạo algorithms dễ dàng
def create_uninformed_search(algorithm: str, verbose: bool = False):
    """
    Factory function để tạo uninformed search algorithms.
    
    Args:
        algorithm: Tên thuật toán ('bfs', 'dfs', 'ucs')
        verbose: In thông tin debug
        
    Returns:
        Instance của thuật toán tương ứng
        
    Example:
        >>> bfs = create_uninformed_search('bfs')
        >>> result = bfs.search(graph, start, goal)
    """
    algorithms = {
        'bfs': BreadthFirstSearch,
        'dfs': DepthFirstSearch,
        'ucs': UniformCostSearch
    }
    
    if algorithm.lower() not in algorithms:
        raise ValueError(f"Unknown algorithm: {algorithm}. "
                        f"Choose from {list(algorithms.keys())}")
    
    return algorithms[algorithm.lower()](verbose=verbose)