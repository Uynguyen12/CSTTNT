"""
Nature-Inspired Algorithms Package
Evolution-based, Physics-based, Swarm-based, Human-based algorithms
"""

# Evolution-based algorithms
from .evolution_based import (
    GeneticAlgorithm,
    DifferentialEvolution
)

# Physics-based algorithms
from .physics_based import (
    GravitationalSearchAlgorithm,
    HarmonySearch,
    SimulatedAnnealing,
    HillClimbing
)

# Swarm/Biology-based algorithms
from .swarm_based import (
    ParticleSwarmOptimization,
    AntColonyOptimization,
    ArtificialBeeColony,
    FireflyAlgorithm,
    CuckooSearch
)

# Human-based algorithms
from .human_based import (
    TeachingLearningBasedOptimization
)

__all__ = [
    # Evolution-based
    'GeneticAlgorithm',
    'DifferentialEvolution',
    
    # Physics-based
    'GravitationalSearchAlgorithm',
    'HarmonySearch',
    'SimulatedAnnealing',
    'HillClimbing',
    
    # Swarm/Biology-based
    'ParticleSwarmOptimization',
    'AntColonyOptimization',
    'ArtificialBeeColony',
    'FireflyAlgorithm',
    'CuckooSearch',
    
    # Human-based
    'TeachingLearningBasedOptimization',
]

__version__ = '1.0.0'


# Factory functions để tạo algorithms dễ dàng

def create_evolution_algorithm(algorithm: str, **kwargs):
    """
    Factory function để tạo evolution-based algorithms.
    
    Args:
        algorithm: Tên thuật toán ('ga', 'de')
        **kwargs: Parameters cho thuật toán
        
    Returns:
        Instance của thuật toán tương ứng
        
    Example:
        >>> ga = create_evolution_algorithm('ga', population_size=100)
        >>> de = create_evolution_algorithm('de', mutation_factor=0.8)
    """
    algorithms = {
        'ga': GeneticAlgorithm,
        'de': DifferentialEvolution
    }
    
    if algorithm.lower() not in algorithms:
        raise ValueError(f"Unknown algorithm: {algorithm}. "
                        f"Choose from {list(algorithms.keys())}")
    
    return algorithms[algorithm.lower()](**kwargs)


def create_physics_algorithm(algorithm: str, **kwargs):
    """
    Factory function để tạo physics-based algorithms.
    
    Args:
        algorithm: Tên thuật toán ('gsa', 'hs')
        **kwargs: Parameters cho thuật toán
        
    Returns:
        Instance của thuật toán tương ứng
        
    Example:
        >>> gsa = create_physics_algorithm('gsa', population_size=50)
        >>> hs = create_physics_algorithm('hs', hmcr=0.9)
    """
    algorithms = {
        'gsa': GravitationalSearchAlgorithm,
        'hs': HarmonySearch
    }
    
    if algorithm.lower() not in algorithms:
        raise ValueError(f"Unknown algorithm: {algorithm}. "
                        f"Choose from {list(algorithms.keys())}")
    
    return algorithms[algorithm.lower()](**kwargs)


def create_swarm_algorithm(algorithm: str, **kwargs):
    """
    Factory function để tạo swarm-based algorithms.
    
    Args:
        algorithm: Tên thuật toán ('pso', 'aco', 'abc', 'fa', 'cs')
        **kwargs: Parameters cho thuật toán
        
    Returns:
        Instance của thuật toán tương ứng
        
    Example:
        >>> pso = create_swarm_algorithm('pso', population_size=50)
        >>> aco = create_swarm_algorithm('aco', archive_size=50)
        >>> abc = create_swarm_algorithm('abc', limit=100)
    """
    algorithms = {
        'pso': ParticleSwarmOptimization,
        'aco': AntColonyOptimization,
        'abc': ArtificialBeeColony,
        'fa': FireflyAlgorithm,
        'cs': CuckooSearch
    }
    
    if algorithm.lower() not in algorithms:
        raise ValueError(f"Unknown algorithm: {algorithm}. "
                        f"Choose from {list(algorithms.keys())}")
    
    return algorithms[algorithm.lower()](**kwargs)


def create_human_algorithm(algorithm: str, **kwargs):
    """
    Factory function để tạo human-based algorithms.
    
    Args:
        algorithm: Tên thuật toán ('tlbo')
        **kwargs: Parameters cho thuật toán
        
    Returns:
        Instance của thuật toán tương ứng
        
    Example:
        >>> tlbo = create_human_algorithm('tlbo', population_size=50)
    """
    algorithms = {
        'tlbo': TeachingLearningBasedOptimization
    }
    
    if algorithm.lower() not in algorithms:
        raise ValueError(f"Unknown algorithm: {algorithm}. "
                        f"Choose from {list(algorithms.keys())}")
    
    return algorithms[algorithm.lower()](**kwargs)