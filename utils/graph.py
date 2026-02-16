import numpy as np
from typing import List, Tuple
import math

class Graph:

    def __init__(self, grid):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.num_nodes = self.rows * self.cols

    def coord_to_id(self, row, col):
        return row * self.cols + col

    def id_to_coord(self, node_id):
        row = node_id // self.cols
        col = node_id % self.cols
        return row, col

    def in_bounds(self, node_id):
        return 0 <= node_id < self.num_nodes

    def passable(self, node_id):
        row, col = self.id_to_coord(node_id)
        return self.grid[row, col] == 0

    def get_neighbors(self, node_id):
        row, col = self.id_to_coord(node_id)
        neighbors = []

        directions = [(-1,0),(1,0),(0,-1),(0,1)]

        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if self.grid[nr, nc] == 0:
                    nid = self.coord_to_id(nr, nc)
                    neighbors.append((nid, 1.0))

        return neighbors

    def euclidean_distance(self, node1, node2):
        r1, c1 = self.id_to_coord(node1)
        r2, c2 = self.id_to_coord(node2)
        return math.sqrt((r1 - r2)**2 + (c1 - c2)**2)

    def manhattan_distance(self, node1, node2):
        r1, c1 = self.id_to_coord(node1)
        r2, c2 = self.id_to_coord(node2)
        return abs(r1 - r2) + abs(c1 - c2)


# =========================================
# Factory function
# =========================================

def create_grid_graph(grid: np.ndarray) -> Graph:
    return Graph(grid)
