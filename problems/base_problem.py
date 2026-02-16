from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Optional
import numpy as np

class BaseProblem(ABC):
    """
    Lớp cơ sở cho tất cả các bài toán tối ưu hóa.
    """
    def __init__(self, name: str, dimensions: int, is_discrete: bool = False):
        self.name = name
        self.dimensions = dimensions
        self.is_discrete = is_discrete

    @abstractmethod
    def evaluate(self, solution: np.ndarray) -> float:
        """Tính toán fitness của một solution."""
        pass

    @abstractmethod
    def get_bounds(self) -> List[Tuple[float, float]]:
        """Trả về giới hạn (lower, upper) cho mỗi chiều."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi bài toán thành dictionary để tương thích với _initialize của Optimizer.
        """
        bounds = self.get_bounds()
        return {
            'objective_func': self.evaluate,
            'dimensions': self.dimensions,
            'lower_bound': bounds[0][0] if bounds else -100, # Lấy bound chiều đầu tiên làm đại diện
            'upper_bound': bounds[0][1] if bounds else 100,
            'is_discrete': self.is_discrete,
            'bounds': bounds # Truyền full bounds nếu cần xử lý chi tiết
        }