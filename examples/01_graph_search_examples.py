"""
Example 1: Graph Search Algorithms
Ví dụ chạy các thuật toán Classical Search trên đồ thị
BFS, DFS, UCS, A*, Greedy Best-First Search
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.graph import Graph
from algorithms.classical.uninformed import BreadthFirstSearch, DepthFirstSearch, UniformCostSearch
from algorithms.classical.informed import AStarSearch, GreedyBestFirstSearch


def create_sample_graph():
    """
    Tạo một sample graph để test
    Graph structure:
           S(0)
          / | \
         1  2  3
        /   |   \
       4    G(5) 6
    """
    graph = Graph(num_nodes=7)
    
    # Add edges with weights
    edges = [
        (0, 1, 1.0),  # S -> 1
        (0, 2, 4.0),  # S -> 2
        (0, 3, 7.0),  # S -> 3
        (1, 4, 3.0),  # 1 -> 4
        (2, 5, 2.0),  # 2 -> G
        (3, 6, 1.0),  # 3 -> 6
        (4, 5, 5.0),  # 4 -> G
        (6, 5, 2.0),  # 6 -> G
    ]
    
    for u, v, weight in edges:
        graph.add_edge(u, v, weight)
    
    return graph


def heuristic_function(node, goal):
    """Simple heuristic: Euclidean distance approximation"""
    # Simplified heuristic values (precomputed)
    heuristics = {
        0: 6,  # S -> G
        1: 5,
        2: 2,
        3: 3,
        4: 5,
        5: 0,  # Goal
        6: 2,
    }
    return heuristics.get(node, 0)


def main():
    print("=" * 60)
    print("GRAPH SEARCH ALGORITHMS - Classical Approach")
    print("=" * 60)
    
    # Create sample graph
    graph = create_sample_graph()
    start = 0  # Node S
    goal = 5   # Node G
    
    print(f"\nGraph: {graph.num_nodes} nodes")
    print(f"Start: Node {start}, Goal: Node {goal}\n")
    
    # ============= Uninformed Search Algorithms =============
    print("\n" + "=" * 60)
    print("UNINFORMED SEARCH ALGORITHMS")
    print("=" * 60)
    
    # 1. Breadth-First Search (BFS)
    print("\n1. Breadth-First Search (BFS)")
    print("-" * 40)
    bfs = BreadthFirstSearch(verbose=True)
    result_bfs = bfs.search(graph, start, goal)
    print(f"Result: {result_bfs}")
    print(f"Path: {result_bfs.path}")
    print(f"Cost: {result_bfs.cost}")
    
    # 2. Depth-First Search (DFS)
    print("\n2. Depth-First Search (DFS)")
    print("-" * 40)
    dfs = DepthFirstSearch(verbose=True)
    result_dfs = dfs.search(graph, start, goal)
    print(f"Result: {result_dfs}")
    print(f"Path: {result_dfs.path}")
    print(f"Cost: {result_dfs.cost}")
    
    # 3. Uniform Cost Search (UCS)
    print("\n3. Uniform Cost Search (UCS)")
    print("-" * 40)
    ucs = UniformCostSearch(verbose=True)
    result_ucs = ucs.search(graph, start, goal)
    print(f"Result: {result_ucs}")
    print(f"Path: {result_ucs.path}")
    print(f"Cost: {result_ucs.cost}")
    
    # ============= Informed Search Algorithms =============
    print("\n" + "=" * 60)
    print("INFORMED SEARCH ALGORITHMS")
    print("=" * 60)
    
    # 4. A* Search
    print("\n4. A* Search")
    print("-" * 40)
    astar = AStarSearch(heuristic=heuristic_function, verbose=True)
    result_astar = astar.search(graph, start, goal)
    print(f"Result: {result_astar}")
    print(f"Path: {result_astar.path}")
    print(f"Cost: {result_astar.cost}")
    
    # 5. Greedy Best-First Search
    print("\n5. Greedy Best-First Search")
    print("-" * 40)
    greedy = GreedyBestFirstSearch(heuristic=heuristic_function, verbose=True)
    result_greedy = greedy.search(graph, start, goal)
    print(f"Result: {result_greedy}")
    print(f"Path: {result_greedy.path}")
    print(f"Cost: {result_greedy.cost}")
    
    # ============= Comparison =============
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    results = {
        "BFS": result_bfs,
        "DFS": result_dfs,
        "UCS": result_ucs,
        "A*": result_astar,
        "Greedy": result_greedy,
    }
    
    print("\n{:<15} {:<15} {:<15} {:<15}".format(
        "Algorithm", "Path Found", "Cost", "Nodes Explored"
    ))
    print("-" * 60)
    for name, result in results.items():
        print("{:<15} {:<15} {:<15.2f} {:<15}".format(
            name,
            "✓" if result.path_found else "✗",
            result.cost,
            result.nodes_explored
        ))


if __name__ == "__main__":
    main()
