"""
Graph Data Structure
Đồ thị cho classical search algorithms
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


class Graph:
    """
    Graph data structure hỗ trợ weighted edges và node positions.
    
    Attributes:
        num_nodes: Số lượng nodes
        adj_matrix: Ma trận kề (adjacency matrix)
        adj_list: Danh sách kề (adjacency list) - dictionary
        node_positions: Vị trí nodes trong không gian 2D (cho heuristics)
    """
    
    def __init__(self, num_nodes: int):
        """
        Initialize graph với số nodes cho trước.
        
        Args:
            num_nodes: Số lượng nodes trong graph
        """
        self.num_nodes = num_nodes
        
        # Adjacency matrix: adj_matrix[u][v] = weight of edge u->v
        # np.inf nghĩa là không có cạnh
        self.adj_matrix = np.full((num_nodes, num_nodes), np.inf)
        np.fill_diagonal(self.adj_matrix, 0)  # Distance to self = 0
        
        # Adjacency list: {node: [(neighbor, weight), ...]}
        self.adj_list: Dict[int, List[Tuple[int, float]]] = {
            i: [] for i in range(num_nodes)
        }
        
        # Node positions: {node: (x, y)}
        self.node_positions: Dict[int, Tuple[float, float]] = {}
    
    def add_edge(self, 
                u: int, 
                v: int, 
                weight: float = 1.0,
                directed: bool = False) -> None:
        """
        Thêm cạnh vào graph.
        
        Args:
            u: Node nguồn
            v: Node đích
            weight: Trọng số của cạnh (default = 1.0)
            directed: Có phải directed edge không (default = False)
        
        Raises:
            ValueError: Nếu u hoặc v nằm ngoài range
        """
        # Validate nodes
        if u < 0 or u >= self.num_nodes or v < 0 or v >= self.num_nodes:
            raise ValueError(f"Invalid nodes: u={u}, v={v}. "
                           f"Valid range: [0, {self.num_nodes-1}]")
        
        # Add edge u -> v
        self.adj_matrix[u][v] = weight
        self.adj_list[u].append((v, weight))
        
        # Add edge v -> u nếu undirected
        if not directed:
            self.adj_matrix[v][u] = weight
            self.adj_list[v].append((u, weight))
    
    def remove_edge(self, u: int, v: int, directed: bool = False) -> None:
        """
        Xóa cạnh khỏi graph.
        
        Args:
            u: Node nguồn
            v: Node đích
            directed: Có phải directed edge không
        """
        # Remove from adjacency matrix
        self.adj_matrix[u][v] = np.inf
        
        # Remove from adjacency list
        self.adj_list[u] = [(n, w) for n, w in self.adj_list[u] if n != v]
        
        if not directed:
            self.adj_matrix[v][u] = np.inf
            self.adj_list[v] = [(n, w) for n, w in self.adj_list[v] if n != u]
    
    def get_neighbors(self, node: int) -> List[Tuple[int, float]]:
        """
        Lấy danh sách neighbors của một node.
        
        Args:
            node: Node ID
            
        Returns:
            List of (neighbor_id, edge_weight)
        """
        if node < 0 or node >= self.num_nodes:
            raise ValueError(f"Invalid node: {node}")
        
        return self.adj_list[node]
    
    def set_node_position(self, node: int, x: float, y: float) -> None:
        """
        Đặt vị trí cho node (để tính heuristic).
        
        Args:
            node: Node ID
            x: Tọa độ x
            y: Tọa độ y
        """
        if node < 0 or node >= self.num_nodes:
            raise ValueError(f"Invalid node: {node}")
        
        self.node_positions[node] = (x, y)
    
    def get_node_position(self, node: int) -> Optional[Tuple[float, float]]:
        """
        Lấy vị trí của node.
        
        Args:
            node: Node ID
            
        Returns:
            (x, y) tuple hoặc None nếu chưa set position
        """
        return self.node_positions.get(node)
    
    def euclidean_distance(self, node1: int, node2: int) -> float:
        """
        Tính Euclidean distance giữa 2 nodes.
        
        Args:
            node1: Node thứ nhất
            node2: Node thứ hai
            
        Returns:
            Euclidean distance, hoặc 0.0 nếu không có position
        """
        if node1 not in self.node_positions or node2 not in self.node_positions:
            return 0.0
        
        x1, y1 = self.node_positions[node1]
        x2, y2 = self.node_positions[node2]
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    
    def manhattan_distance(self, node1: int, node2: int) -> float:
        """
        Tính Manhattan distance giữa 2 nodes.
        
        Args:
            node1: Node thứ nhất
            node2: Node thứ hai
            
        Returns:
            Manhattan distance, hoặc 0.0 nếu không có position
        """
        if node1 not in self.node_positions or node2 not in self.node_positions:
            return 0.0
        
        x1, y1 = self.node_positions[node1]
        x2, y2 = self.node_positions[node2]
        return abs(x1 - x2) + abs(y1 - y2)
    
    def has_edge(self, u: int, v: int) -> bool:
        """Check xem có cạnh u -> v không."""
        return self.adj_matrix[u][v] != np.inf
    
    def get_edge_weight(self, u: int, v: int) -> float:
        """Lấy trọng số của cạnh u -> v."""
        return self.adj_matrix[u][v]
    
    def get_degree(self, node: int) -> int:
        """Lấy degree (số neighbors) của node."""
        return len(self.adj_list[node])
    
    def is_connected(self) -> bool:
        """
        Check xem graph có connected không (dùng BFS).
        
        Returns:
            True nếu tất cả nodes đều reachable từ node 0
        """
        if self.num_nodes == 0:
            return True
        
        visited = np.zeros(self.num_nodes, dtype=bool)
        queue = [0]
        visited[0] = True
        count = 1
        
        while queue:
            current = queue.pop(0)
            for neighbor, _ in self.adj_list[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
                    count += 1
        
        return count == self.num_nodes
    
    def __repr__(self) -> str:
        return (f"Graph(nodes={self.num_nodes}, "
                f"edges={self._count_edges()}, "
                f"directed={self._is_directed()})")
    
    def _count_edges(self) -> int:
        """Đếm số cạnh trong graph."""
        count = 0
        for neighbors in self.adj_list.values():
            count += len(neighbors)
        
        # Nếu undirected, mỗi cạnh được count 2 lần
        if not self._is_directed():
            count //= 2
        
        return count
    
    def _is_directed(self) -> bool:
        """Check xem graph có directed không."""
        # Simple heuristic: nếu có cạnh u->v mà không có v->u thì directed
        for u in range(self.num_nodes):
            for v, _ in self.adj_list[u]:
                if not any(n == u for n, _ in self.adj_list[v]):
                    return True
        return False
    
    def to_dict(self) -> dict:
        """
        Convert graph thành dictionary format (cho serialization).
        
        Returns:
            Dictionary chứa thông tin graph
        """
        return {
            'num_nodes': self.num_nodes,
            'edges': [
                (u, v, w) 
                for u in range(self.num_nodes)
                for v, w in self.adj_list[u]
            ],
            'positions': self.node_positions
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Graph':
        """
        Tạo graph từ dictionary.
        
        Args:
            data: Dictionary chứa thông tin graph
            
        Returns:
            Graph object
        """
        graph = cls(data['num_nodes'])
        
        for u, v, w in data['edges']:
            graph.add_edge(u, v, w, directed=True)
        
        for node, (x, y) in data['positions'].items():
            graph.set_node_position(int(node), x, y)
        
        return graph


def create_grid_graph(rows: int, cols: int) -> Graph:
    """
    Tạo grid graph (maze-like).
    
    Args:
        rows: Số hàng
        cols: Số cột
        
    Returns:
        Graph object với structure dạng grid
    """
    num_nodes = rows * cols
    graph = Graph(num_nodes)
    
    def to_node(r: int, c: int) -> int:
        return r * cols + c
    
    # Add edges giữa các cells kề nhau
    for r in range(rows):
        for c in range(cols):
            node = to_node(r, c)
            
            # Right neighbor
            if c + 1 < cols:
                graph.add_edge(node, to_node(r, c + 1), 1.0, directed=False)
            
            # Down neighbor
            if r + 1 < rows:
                graph.add_edge(node, to_node(r + 1, c), 1.0, directed=False)
    
    # Set positions
    for r in range(rows):
        for c in range(cols):
            node = to_node(r, c)
            graph.set_node_position(node, c, rows - r - 1)
    
    return graph