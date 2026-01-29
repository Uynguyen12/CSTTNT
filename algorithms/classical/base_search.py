"""
Base Search Algorithm Classes
Abstract classes cho graph search algorithms
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, TYPE_CHECKING
import numpy as np
import time

if TYPE_CHECKING:
    from utils.graph import Graph


@dataclass
class SearchResults:
    """
    Container cho kết quả của graph search algorithms.
    
    Attributes:
        path: Đường đi từ start đến goal (list of node IDs)
        cost: Tổng chi phí của đường đi
        path_found: Có tìm thấy đường đi không
        nodes_explored: Số nodes đã explore
        execution_time: Thời gian thực thi (seconds)
        explored_order: Thứ tự explore các nodes
        metadata: Thông tin bổ sung
    """
    path: List[int] = field(default_factory=list)
    cost: float = 0.0
    path_found: bool = False
    nodes_explored: int = 0
    execution_time: float = 0.0
    explored_order: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        status = "✓ Found" if self.path_found else "✗ Not Found"
        return (f"SearchResults({status}, "
                f"cost={self.cost:.2f}, "
                f"nodes={self.nodes_explored}, "
                f"time={self.execution_time*1000:.3f}ms)")


class BaseSearchAlgorithm(ABC):
    """
    Abstract base class cho tất cả graph search algorithms.
    
    Subclasses phải implement:
        - _search(): Logic chính của thuật toán
    """
    
    def __init__(self, verbose: bool = False):
        """
        Args:
            verbose: In thông tin debug không
        """
        self.verbose = verbose
        self._start_time = 0.0
        self._nodes_explored = 0
        self._explored_order = []
    
    @abstractmethod
    def _search(self, 
               graph: 'Graph',
               start: int,
               goal: int) -> SearchResults:
        """
        Thuật toán tìm kiếm chính.
        
        Args:
            graph: Graph object
            start: Node bắt đầu
            goal: Node đích
            
        Returns:
            SearchResults object
        """
        pass
    
    def search(self, 
              graph: 'Graph',
              start: int,
              goal: int) -> SearchResults:
        """
        Public method để chạy search algorithm.
        
        Wrapper around _search() để thêm timing và validation.
        """
        # Validate inputs
        self._validate_inputs(graph, start, goal)
        
        # Reset state
        self._nodes_explored = 0
        self._explored_order = []
        
        # Run search
        self._start_time = time.perf_counter()
        result = self._search(graph, start, goal)
        result.execution_time = time.perf_counter() - self._start_time
        
        # Add metadata
        result.metadata['algorithm'] = self.__class__.__name__
        result.metadata['graph_size'] = graph.num_nodes
        
        if self.verbose:
            self._print_result(result)
        
        return result
    
    def _validate_inputs(self, graph: 'Graph', start: int, goal: int) -> None:
        """Validate inputs trước khi search."""
        if start < 0 or start >= graph.num_nodes:
            raise ValueError(f"Invalid start node: {start}")
        if goal < 0 or goal >= graph.num_nodes:
            raise ValueError(f"Invalid goal node: {goal}")
    
    def _reconstruct_path(self, 
                         parent: np.ndarray,
                         start: int,
                         goal: int) -> List[int]:
        """
        Reconstruct path từ parent pointers.
        
        Args:
            parent: Array of parent nodes
            start: Start node
            goal: Goal node
            
        Returns:
            path: List of nodes from start to goal
        """
        path = []
        current = goal
        
        while current != -1:
            path.append(int(current))
            if current == start:
                break
            current = parent[current]
        
        return list(reversed(path))
    
    def _print_result(self, result: SearchResults) -> None:
        """In kết quả search."""
        print(f"\n{'='*70}")
        print(f"Algorithm: {self.__class__.__name__}")
        print(f"{'='*70}")
        
        if result.path_found:
            print(f"✓ Path found!")
            print(f"  Path: {' → '.join(map(str, result.path))}")
            print(f"  Cost: {result.cost:.2f}")
        else:
            print(f"✗ No path found")
        
        print(f"  Nodes explored: {result.nodes_explored}")
        print(f"  Execution time: {result.execution_time*1000:.3f} ms")
        print(f"{'='*70}")


class UninformedSearchAlgorithm(BaseSearchAlgorithm):
    """
    Base class cho Uninformed Search algorithms (BFS, DFS, UCS).
    
    Không sử dụng heuristic information.
    """
    
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)


class InformedSearchAlgorithm(BaseSearchAlgorithm):
    """
    Base class cho Informed Search algorithms (Greedy, A*).
    
    Sử dụng heuristic function để guide search.
    """
    
    def __init__(self, 
                 heuristic = 'euclidean',
                 verbose: bool = False):
        """
        Args:
            heuristic: Loại heuristic (str: 'euclidean'/'manhattan') hoặc callable function
            verbose: In thông tin debug
        """
        super().__init__(verbose)
        self.heuristic = heuristic
        self._heuristic_callable = None
        
        # Nếu heuristic là callable, lưu nó
        if callable(heuristic):
            self._heuristic_callable = heuristic
    
    def _get_heuristic_function(self, graph: 'Graph') -> Callable:
        """
        Trả về heuristic function dựa trên loại heuristic.
        
        Args:
            graph: Graph object
            
        Returns:
            heuristic_func: Function(node1, node2) -> float
        """
        # Nếu đã có callable heuristic, dùng nó
        if self._heuristic_callable is not None:
            return self._heuristic_callable
        
        # Nếu heuristic là string, tìm loại tương ứng
        if isinstance(self.heuristic, str):
            if self.heuristic == 'euclidean':
                return graph.euclidean_distance
            elif self.heuristic == 'manhattan':
                return graph.manhattan_distance
            else:
                raise ValueError(f"Unknown heuristic: {self.heuristic}")
        else:
            raise ValueError(f"Heuristic must be string or callable, got {type(self.heuristic)}")
    
    def _compute_heuristic(self,
                          graph: 'Graph',
                          node: int,
                          goal: int) -> float:
        """
        Tính heuristic value từ node đến goal.
        
        Args:
            graph: Graph object
            node: Current node
            goal: Goal node
            
        Returns:
            h_value: Heuristic estimate
        """
        h_func = self._get_heuristic_function(graph)
        return h_func(node, goal)


class LocalSearchAlgorithm(BaseSearchAlgorithm):
    """
    Base class cho Local Search algorithms (Hill Climbing, Simulated Annealing).
    
    Không maintain frontier, chỉ explore local neighborhood.
    """
    
    def __init__(self,
                 heuristic: str = 'euclidean',
                 verbose: bool = False):
        """
        Args:
            heuristic: Loại heuristic
            verbose: In thông tin debug
        """
        super().__init__(verbose)
        self.heuristic = heuristic
    
    def _get_heuristic_function(self, graph: 'Graph') -> Callable:
        """Get heuristic function (giống InformedSearchAlgorithm)."""
        if self.heuristic == 'euclidean':
            return graph.euclidean_distance
        elif self.heuristic == 'manhattan':
            return graph.manhattan_distance
        else:
            raise ValueError(f"Unknown heuristic: {self.heuristic}")
    
    def _compute_heuristic(self, 
                          graph: 'Graph',
                          node: int, 
                          goal: int) -> float:
        """Compute heuristic value."""
        h_func = self._get_heuristic_function(graph)
        return h_func(node, goal)
