import numpy as np
from typing import List, Tuple, Optional
from .base_problem import BaseProblem

class TSP(BaseProblem):
    """
    Traveling Salesman Problem (TSP).
    Mục tiêu: Tìm chu trình ngắn nhất đi qua tất cả thành phố.
    Solution Encoding: Hoán vị (Permutation) của các index thành phố.
    """
    def __init__(self, num_cities: int, city_coords: Optional[np.ndarray] = None, seed: int = 42):
        super().__init__("TSP", num_cities, is_discrete=True)
        
        np.random.seed(seed)
        if city_coords is None:
            # Tạo tọa độ ngẫu nhiên [0, 100] x [0, 100]
            self.city_coords = np.random.rand(num_cities, 2) * 100
        else:
            self.city_coords = city_coords
            
        # Tính trước ma trận khoảng cách để tăng tốc độ evaluate
        self.distance_matrix = self._calculate_distance_matrix()

    def _calculate_distance_matrix(self) -> np.ndarray:
        num_cities = len(self.city_coords)
        dist_matrix = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    dist = np.linalg.norm(self.city_coords[i] - self.city_coords[j])
                    dist_matrix[i][j] = dist
        return dist_matrix

    def get_bounds(self) -> List[Tuple[float, float]]:
        # Với TSP dùng permutation, bounds không thực sự quan trọng theo nghĩa min/max giá trị
        # Nhưng để tương thích interface, ta trả về range index
        return [(0, self.dimensions - 1)] * self.dimensions

    def evaluate(self, solution: np.ndarray) -> float:
        """
        Tính tổng độ dài đường đi.
        solution: Mảng các index thành phố (VD: [0, 2, 1, 3])
        """
        # Đảm bảo solution là integer để index
        route = solution.astype(int)
        total_distance = 0.0
        
        for i in range(len(route) - 1):
            from_city = route[i]
            to_city = route[i + 1]
            total_distance += self.distance_matrix[from_city][to_city]
            
        # Cộng khoảng cách quay về điểm đầu
        total_distance += self.distance_matrix[route[-1]][route[0]]
        
        return total_distance