"""
Classical Search Algorithms Package
BFS, DFS, UCS, Greedy, A*, Hill Climbing, SA
"""

from .base_search import (
    BaseSearchAlgorithm,
    UninformedSearchAlgorithm,
    InformedSearchAlgorithm,
    LocalSearchAlgorithm,
    SearchResults
)

from .uninformed import (
    BreadthFirstSearch,
    DepthFirstSearch,
    UniformCostSearch,
    create_uninformed_search
)

from .informed import (
    GreedyBestFirstSearch,
    AStarSearch,
    create_informed_search
)

from .local_search import (
    HillClimbingSearch,
    SimulatedAnnealingSearch,
    create_local_search
)

__all__ = [
    # Base classes
    'BaseSearchAlgorithm',
    'UninformedSearchAlgorithm',
    'InformedSearchAlgorithm',
    'LocalSearchAlgorithm',
    'SearchResults',
    
    # Uninformed algorithms
    'BreadthFirstSearch',
    'DepthFirstSearch',
    'UniformCostSearch',
    'create_uninformed_search',
    
    # Informed algorithms
    'GreedyBestFirstSearch',
    'AStarSearch',
    'create_informed_search',
    
    # Local search algorithms
    'HillClimbingSearch',
    'SimulatedAnnealingSearch',
    'create_local_search',
]

__version__ = '1.0.0'