"""
Example: Demonstrating Classical Search Algorithms

Hướng dẫn sử dụng tất cả các thuật toán search được implement.
"""

from algorithms.classical import (
    # Uninformed Search
    BreadthFirstSearch,
    DepthFirstSearch,
    UniformCostSearch,
    create_uninformed_search,
    
    # Informed Search
    GreedyBestFirstSearch,
    AStarSearch,
    create_informed_search,
    
    # Local Search
    HillClimbingSearch,
    SimulatedAnnealingSearch,
    create_local_search
)

from utils.graph import Graph


def create_sample_graph():
    """Tạo sample graph để test algorithms."""
    # Create graph với 6 nodes
    graph = Graph(num_nodes=6)
    
    # Set node positions cho heuristics
    positions = {
        0: (0, 0),      # Start
        1: (1, 1),
        2: (1, -1),
        3: (2, 1),
        4: (2, -1),
        5: (3, 0)       # Goal
    }
    graph.node_positions = positions
    
    # Add edges (undirected)
    edges = [
        (0, 1, 1.0),
        (0, 2, 1.0),
        (1, 3, 1.5),
        (2, 4, 1.5),
        (1, 2, 2.0),
        (3, 5, 1.0),
        (4, 5, 1.0),
    ]
    
    for u, v, weight in edges:
        graph.add_edge(u, v, weight=weight, directed=False)
    
    return graph


def compare_uninformed_algorithms(graph, start=0, goal=5):
    """So sánh các Uninformed Search algorithms."""
    print("\n" + "="*70)
    print("UNINFORMED SEARCH ALGORITHMS")
    print("="*70)
    
    algorithms = ['bfs', 'dfs', 'ucs']
    
    for algo_name in algorithms:
        algo = create_uninformed_search(algo_name, verbose=True)
        result = algo.search(graph, start=start, goal=goal)
        print(f"\nExplored order: {result.explored_order}")
        print(result)


def compare_informed_algorithms(graph, start=0, goal=5):
    """So sánh các Informed Search algorithms."""
    print("\n" + "="*70)
    print("INFORMED SEARCH ALGORITHMS")
    print("="*70)
    
    algorithms = ['greedy', 'astar']
    heuristics = ['euclidean', 'manhattan']
    
    for algo_name in algorithms:
        for heur in heuristics:
            print(f"\n--- {algo_name.upper()} with {heur} heuristic ---")
            algo = create_informed_search(algo_name, heuristic=heur, verbose=True)
            result = algo.search(graph, start=start, goal=goal)
            print(f"Explored order: {result.explored_order}")
            print(result)


def compare_local_search_algorithms(graph, start=0, goal=5):
    """So sánh các Local Search algorithms."""
    print("\n" + "="*70)
    print("LOCAL SEARCH ALGORITHMS")
    print("="*70)
    
    # Hill Climbing
    print("\n--- HILL CLIMBING ---")
    hc = create_local_search('hillclimbing', heuristic='euclidean', verbose=True)
    result_hc = hc.search(graph, start=start, goal=goal)
    print(f"Explored order: {result_hc.explored_order}")
    print(result_hc)
    
    # Simulated Annealing
    print("\n--- SIMULATED ANNEALING ---")
    sa = create_local_search(
        'simulated_annealing',
        heuristic='euclidean',
        max_iterations=5000,
        temperature=100.0,
        cooling_rate=0.95,
        verbose=True
    )
    result_sa = sa.search(graph, start=start, goal=goal)
    print(f"Explored order: {result_sa.explored_order}")
    print(result_sa)


def main():
    """Main function."""
    print("\n" + "="*70)
    print("CLASSICAL SEARCH ALGORITHMS - COMPREHENSIVE EXAMPLE")
    print("="*70)
    
    # Create sample graph
    graph = create_sample_graph()
    print(f"\nGraph: {graph.num_nodes} nodes, start=0, goal=5")
    print(f"Node positions: {graph.node_positions}")
    
    # Test all algorithms
    compare_uninformed_algorithms(graph)
    compare_informed_algorithms(graph)
    compare_local_search_algorithms(graph)
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
