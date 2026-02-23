"""
Test Cases for Discrete Optimization Problems
Suitable for ACO, PSO, ABC, FA, CS, GA

Bài toán:
1. Traveling Salesman Problem (TSP) - lớn (50-100 cities)
2. Knapsack Problem (KP) - lớn (100-200 items)
3. Graph Coloring Problem (GC)
"""

import numpy as np
from typing import List, Tuple, Dict
import pickle


class TSPInstance:
    """
    Traveling Salesman Problem Instance
    
    Với constraints về time và cost:
    - Mỗi city có time window
    - Có giới hạn về tổng thời gian
    - Có nhiều cost factors (distance, tolls, etc.)
    """
    
    def __init__(self, num_cities: int, seed: int = 42):
        """
        Tạo TSP instance.
        
        Args:
            num_cities: Số lượng cities (50-100 recommended)
            seed: Random seed
        """
        np.random.seed(seed)
        self.num_cities = num_cities
        
        # Depot (starting city)
        self.depot = 0
        
        # Generate city positions (normalized to [0, 100])
        self.positions = np.random.rand(num_cities, 2) * 100
        
        # Calculate distance matrix
        self.distance_matrix = self._calculate_distances()
        
        # Time windows for each city (start_time, end_time)
        self.time_windows = self._generate_time_windows()
        
        # Service time at each city
        self.service_times = np.random.uniform(0.5, 2.0, num_cities)
        
        # Cost factors (distance cost, time cost, penalty for violating windows)
        self.distance_cost_factor = 1.0
        self.time_cost_factor = 0.5
        self.window_violation_penalty = 100.0
        
        # Maximum total time allowed
        self.max_total_time = num_cities * 3.0
    
    def _calculate_distances(self) -> np.ndarray:
        """Calculate Euclidean distance matrix."""
        distances = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                dist = np.linalg.norm(self.positions[i] - self.positions[j])
                distances[i][j] = dist
                distances[j][i] = dist
        return distances
    
    def _generate_time_windows(self) -> List[Tuple[float, float]]:
        """Generate time windows for cities."""
        windows = []
        for i in range(self.num_cities):
            if i == self.depot:
                # Depot is always available
                windows.append((0.0, self.num_cities * 5.0))
            else:
                # Random time window
                start = np.random.uniform(0, self.num_cities * 2)
                duration = np.random.uniform(2, 10)
                windows.append((start, start + duration))
        return windows
    
    def evaluate_tour(self, tour: List[int]) -> float:
        """
        Evaluate cost of a tour.
        
        Args:
            tour: List of city indices (permutation)
            
        Returns:
            Total cost (lower is better)
        """
        if len(tour) != self.num_cities:
            return float('inf')
        
        total_distance = 0.0
        total_time = 0.0
        violations = 0
        current_time = 0.0
        
        # Start from depot
        for i in range(len(tour)):
            current_city = tour[i]
            next_city = tour[(i + 1) % len(tour)]
            
            # Add travel distance
            travel_dist = self.distance_matrix[current_city][next_city]
            total_distance += travel_dist
            
            # Add travel time (assume speed = 1 unit/time)
            travel_time = travel_dist
            current_time += travel_time
            
            # Check time window
            start_window, end_window = self.time_windows[next_city]
            if current_time < start_window:
                # Wait until window opens
                current_time = start_window
            elif current_time > end_window:
                # Violation
                violations += 1
            
            # Add service time
            current_time += self.service_times[next_city]
            total_time += travel_time + self.service_times[next_city]
        
        # Calculate total cost
        cost = (self.distance_cost_factor * total_distance +
                self.time_cost_factor * total_time +
                self.window_violation_penalty * violations)
        
        # Penalty for exceeding max time
        if total_time > self.max_total_time:
            cost += self.window_violation_penalty * (total_time - self.max_total_time)
        
        return cost
    
    def get_nearest_neighbor_solution(self) -> List[int]:
        """
        Generate initial solution using nearest neighbor heuristic.
        
        Returns:
            Tour (list of city indices)
        """
        unvisited = set(range(self.num_cities))
        tour = [self.depot]
        unvisited.remove(self.depot)
        
        current = self.depot
        while unvisited:
            # Find nearest unvisited city
            nearest = min(unvisited, key=lambda c: self.distance_matrix[current][c])
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        return tour
    
    def save(self, filename: str):
        """Save instance to file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filename: str):
        """Load instance from file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)


class KnapsackInstance:
    """
    Knapsack Problem Instance (0/1 Knapsack)
    
    Large-scale với nhiều items (100-200).
    """
    
    def __init__(self, num_items: int, capacity_ratio: float = 0.5, seed: int = 42):
        """
        Tạo Knapsack instance.
        
        Args:
            num_items: Số lượng items (100-200 recommended)
            capacity_ratio: Tỷ lệ capacity so với tổng weight
            seed: Random seed
        """
        np.random.seed(seed)
        self.num_items = num_items
        
        # Generate item weights and values
        self.weights = np.random.randint(1, 50, num_items).astype(float)
        self.values = np.random.randint(1, 100, num_items).astype(float)
        
        # Add correlation between weight and value for some items
        # (makes problem more realistic and harder)
        for i in range(num_items // 3):
            idx = np.random.randint(0, num_items)
            self.values[idx] = self.weights[idx] * np.random.uniform(1.5, 3.0)
        
        # Set capacity
        total_weight = np.sum(self.weights)
        self.capacity = total_weight * capacity_ratio
        
        # Calculate value densities (for heuristics)
        self.value_densities = self.values / self.weights
    
    def evaluate_solution(self, solution: np.ndarray) -> Tuple[float, bool]:
        """
        Evaluate a solution (binary vector).
        
        Args:
            solution: Binary array (1 = include item, 0 = exclude)
            
        Returns:
            (value, is_feasible)
        """
        total_weight = np.sum(solution * self.weights)
        total_value = np.sum(solution * self.values)
        
        is_feasible = total_weight <= self.capacity
        
        if not is_feasible:
            # Penalize infeasible solutions
            penalty = (total_weight - self.capacity) * 1000
            return -penalty, False
        
        return total_value, True
    
    def get_greedy_solution(self) -> np.ndarray:
        """
        Generate initial solution using greedy heuristic.
        Sorted by value density.
        
        Returns:
            Binary solution vector
        """
        # Sort items by value density (descending)
        sorted_indices = np.argsort(-self.value_densities)
        
        solution = np.zeros(self.num_items)
        current_weight = 0.0
        
        for idx in sorted_indices:
            if current_weight + self.weights[idx] <= self.capacity:
                solution[idx] = 1
                current_weight += self.weights[idx]
        
        return solution
    
    def repair_solution(self, solution: np.ndarray) -> np.ndarray:
        """
        Repair infeasible solution by removing items.
        
        Args:
            solution: Binary solution (possibly infeasible)
            
        Returns:
            Repaired feasible solution
        """
        solution = solution.copy()
        total_weight = np.sum(solution * self.weights)
        
        if total_weight <= self.capacity:
            return solution
        
        # Remove items with lowest value density first
        selected_indices = np.where(solution > 0)[0]
        sorted_selected = sorted(selected_indices, 
                                key=lambda i: self.value_densities[i])
        
        for idx in sorted_selected:
            solution[idx] = 0
            total_weight -= self.weights[idx]
            if total_weight <= self.capacity:
                break
        
        return solution
    
    def save(self, filename: str):
        """Save instance to file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filename: str):
        """Load instance from file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)


class GraphColoringInstance:
    """
    Graph Coloring Problem Instance
    
    Given a graph, assign colors to vertices such that
    no two adjacent vertices have the same color.
    Objective: Minimize number of colors.
    """
    
    def __init__(self, num_vertices: int, edge_probability: float = 0.3, seed: int = 42):
        """
        Tạo Graph Coloring instance.
        
        Args:
            num_vertices: Số lượng vertices (50-100 recommended)
            edge_probability: Xác suất có edge giữa 2 vertices
            seed: Random seed
        """
        np.random.seed(seed)
        self.num_vertices = num_vertices
        
        # Generate random graph (adjacency matrix)
        self.adj_matrix = np.random.rand(num_vertices, num_vertices) < edge_probability
        
        # Make symmetric (undirected graph)
        self.adj_matrix = np.logical_or(self.adj_matrix, self.adj_matrix.T)
        
        # No self-loops
        np.fill_diagonal(self.adj_matrix, False)
        
        # Adjacency list
        self.adj_list = [
            np.where(self.adj_matrix[i])[0].tolist()
            for i in range(num_vertices)
        ]
        
        # Calculate chromatic number lower bound (max clique size)
        self.chromatic_lower_bound = self._estimate_chromatic_number()
    
    def _estimate_chromatic_number(self) -> int:
        """Estimate chromatic number (greedy upper bound)."""
        colors = np.full(self.num_vertices, -1)
        
        for vertex in range(self.num_vertices):
            # Get colors of neighbors
            neighbor_colors = set()
            for neighbor in self.adj_list[vertex]:
                if colors[neighbor] != -1:
                    neighbor_colors.add(colors[neighbor])
            
            # Assign smallest available color
            color = 0
            while color in neighbor_colors:
                color += 1
            colors[vertex] = color
        
        return np.max(colors) + 1
    
    def evaluate_coloring(self, coloring: np.ndarray) -> Tuple[int, int]:
        """
        Evaluate a coloring.
        
        Args:
            coloring: Array of colors for each vertex
            
        Returns:
            (num_colors, num_conflicts)
        """
        num_colors = len(np.unique(coloring))
        
        # Count conflicts (adjacent vertices with same color)
        conflicts = 0
        for v in range(self.num_vertices):
            for neighbor in self.adj_list[v]:
                if neighbor > v and coloring[v] == coloring[neighbor]:
                    conflicts += 1
        
        return num_colors, conflicts
    
    def get_greedy_coloring(self) -> np.ndarray:
        """
        Generate coloring using greedy algorithm.
        
        Returns:
            Color assignment array
        """
        coloring = np.full(self.num_vertices, -1)
        
        for vertex in range(self.num_vertices):
            # Get colors of neighbors
            neighbor_colors = set()
            for neighbor in self.adj_list[vertex]:
                if coloring[neighbor] != -1:
                    neighbor_colors.add(coloring[neighbor])
            
            # Assign smallest available color
            color = 0
            while color in neighbor_colors:
                color += 1
            coloring[vertex] = color
        
        return coloring
    
    def save(self, filename: str):
        """Save instance to file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filename: str):
        """Load instance from file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)


# ============= Test Case Generators =============

def generate_tsp_test_cases():
    """Generate TSP test cases of different sizes."""
    sizes = [30, 50, 75, 100]
    test_cases = {}
    
    for size in sizes:
        print(f"Generating TSP instance with {size} cities...")
        instance = TSPInstance(num_cities=size, seed=42)
        test_cases[f'tsp_{size}'] = instance
        
        # Print info
        nn_solution = instance.get_nearest_neighbor_solution()
        nn_cost = instance.evaluate_tour(nn_solution)
        print(f"  Nearest Neighbor cost: {nn_cost:.2f}")
    
    return test_cases


def generate_knapsack_test_cases():
    """Generate Knapsack test cases of different sizes."""
    sizes = [50, 100, 150, 200]
    test_cases = {}
    
    for size in sizes:
        print(f"Generating Knapsack instance with {size} items...")
        instance = KnapsackInstance(num_items=size, capacity_ratio=0.5, seed=42)
        test_cases[f'knapsack_{size}'] = instance
        
        # Print info
        greedy_solution = instance.get_greedy_solution()
        greedy_value, feasible = instance.evaluate_solution(greedy_solution)
        print(f"  Greedy solution value: {greedy_value:.2f} (feasible: {feasible})")
        print(f"  Capacity: {instance.capacity:.2f}")
    
    return test_cases


def generate_graph_coloring_test_cases():
    """Generate Graph Coloring test cases of different sizes."""
    configs = [
        (30, 0.2),
        (50, 0.25),
        (75, 0.3),
        (100, 0.3),
    ]
    test_cases = {}
    
    for num_vertices, edge_prob in configs:
        print(f"Generating Graph Coloring instance: {num_vertices} vertices, "
              f"edge_prob={edge_prob}...")
        instance = GraphColoringInstance(num_vertices, edge_prob, seed=42)
        test_cases[f'gc_{num_vertices}'] = instance
        
        # Print info
        greedy_coloring = instance.get_greedy_coloring()
        num_colors, conflicts = instance.evaluate_coloring(greedy_coloring)
        print(f"  Greedy coloring: {num_colors} colors, {conflicts} conflicts")
        print(f"  Chromatic lower bound: {instance.chromatic_lower_bound}")
    
    return test_cases


def main():
    """Generate all test cases."""
    print("=" * 70)
    print("GENERATING DISCRETE OPTIMIZATION TEST CASES")
    print("=" * 70)
    
    # Generate test cases
    print("\n1. TSP Test Cases")
    print("-" * 70)
    tsp_cases = generate_tsp_test_cases()
    
    print("\n2. Knapsack Test Cases")
    print("-" * 70)
    knapsack_cases = generate_knapsack_test_cases()
    
    print("\n3. Graph Coloring Test Cases")
    print("-" * 70)
    gc_cases = generate_graph_coloring_test_cases()
    
    # Save all test cases
    all_test_cases = {
        'tsp': tsp_cases,
        'knapsack': knapsack_cases,
        'graph_coloring': gc_cases
    }
    
    print("\n" + "=" * 70)
    print("Saving test cases...")
    with open('discrete_test_cases.pkl', 'wb') as f:
        pickle.dump(all_test_cases, f)
    
    print("Test cases saved to 'discrete_test_cases.pkl'")
    print("=" * 70)
    
    return all_test_cases


if __name__ == "__main__":
    test_cases = main()