"""
Simple Example - Test Classical Search Algorithms
"""

from algorithms.classical import (
    create_uninformed_search,
    create_informed_search,
    create_local_search
)
from utils.graph import Graph


def create_sample_graph():
    """Tạo graph mẫu để test."""
    graph = Graph(num_nodes=6)
    
    # Set node positions cho heuristics
    graph.node_positions = {
        0: (0, 0),      # Start
        1: (1, 1),
        2: (1, -1),
        3: (2, 1),
        4: (2, -1),
        5: (3, 0)       # Goal
    }
    
    # Add edges
    graph.add_edge(0, 1, weight=1.0)
    graph.add_edge(0, 2, weight=1.0)
    graph.add_edge(1, 3, weight=1.5)
    graph.add_edge(2, 4, weight=1.5)
    graph.add_edge(1, 2, weight=2.0)
    graph.add_edge(3, 5, weight=1.0)
    graph.add_edge(4, 5, weight=1.0)
    
    return graph


def main():
    print("\n" + "="*60)
    print("CLASSICAL SEARCH ALGORITHMS - DEMO")
    print("="*60)
    
    graph = create_sample_graph()
    start, goal = 0, 5
    
    print(f"\nGraph: {graph.num_nodes} nodes, start={start}, goal={goal}")
    
    # Test BFS
    print("\n--- BFS (Breadth-First Search) ---")
    bfs = create_uninformed_search('bfs', verbose=True)
    result_bfs = bfs.search(graph, start=start, goal=goal)
    print(f"Path: {result_bfs.path}, Cost: {result_bfs.cost:.2f}, Explored: {result_bfs.nodes_explored}")
    
    # Test A* Search
    print("\n--- A* Search (Euclidean) ---")
    astar = create_informed_search('astar', heuristic='euclidean', verbose=True)
    result_astar = astar.search(graph, start=start, goal=goal)
    print(f"Path: {result_astar.path}, Cost: {result_astar.cost:.2f}, Explored: {result_astar.nodes_explored}")
    
    # Test Greedy
    print("\n--- Greedy Best-First Search ---")
    greedy = create_informed_search('greedy', heuristic='euclidean', verbose=True)
    result_greedy = greedy.search(graph, start=start, goal=goal)
    print(f"Path: {result_greedy.path}, Cost: {result_greedy.cost:.2f}, Explored: {result_greedy.nodes_explored}")
    
    # Test Hill Climbing
    print("\n--- Hill Climbing ---")
    hc = create_local_search('hillclimbing', heuristic='euclidean', verbose=True)
    result_hc = hc.search(graph, start=start, goal=goal)
    print(f"Path: {result_hc.path}, Cost: {result_hc.cost:.2f}, Explored: {result_hc.nodes_explored}")
    
    # Test Simulated Annealing
    print("\n--- Simulated Annealing ---")
    sa = create_local_search('simulated_annealing', 
                            heuristic='euclidean',
                            max_iterations=1000,
                            temperature=100.0,
                            cooling_rate=0.95,
                            verbose=True)
    result_sa = sa.search(graph, start=start, goal=goal)
    print(f"Path: {result_sa.path}, Cost: {result_sa.cost:.2f}, Explored: {result_sa.nodes_explored}")
    
    print("\n" + "="*60)
    print("COMPARISON TABLE")
    print("="*60)
    print(f"{'Algorithm':<25} {'Path':<20} {'Cost':<10} {'Explored':<10}")
    print("-" * 60)
    print(f"{'BFS':<25} {str(result_bfs.path):<20} {result_bfs.cost:<10.2f} {result_bfs.nodes_explored:<10}")
    print(f"{'A* (Euclidean)':<25} {str(result_astar.path):<20} {result_astar.cost:<10.2f} {result_astar.nodes_explored:<10}")
    print(f"{'Greedy':<25} {str(result_greedy.path):<20} {result_greedy.cost:<10.2f} {result_greedy.nodes_explored:<10}")
    print(f"{'Hill Climbing':<25} {str(result_hc.path):<20} {result_hc.cost:<10.2f} {result_hc.nodes_explored:<10}")
    print(f"{'Simulated Annealing':<25} {str(result_sa.path):<20} {result_sa.cost:<10.2f} {result_sa.nodes_explored:<10}")
    print("="*60)


if __name__ == "__main__":
    main()
