import numpy as np
from typing import List, Tuple
from .base_problem import BaseProblem

class ContinuousProblem(BaseProblem):
    def __init__(self, name: str, dimensions: int, lower: float, upper: float):
        super().__init__(name, dimensions, is_discrete=False)
        self.lower = lower
        self.upper = upper

    def get_bounds(self) -> List[Tuple[float, float]]:
        return [(self.lower, self.upper)] * self.dimensions

class SphereFunction(ContinuousProblem):
    """
    Hàm Sphere: f(x) = sum(x_i^2)
    Global Minimum: 0 tại x = [0, 0, ..., 0]
    """
    def __init__(self, dimensions: int = 10):
        super().__init__("Sphere Function", dimensions, -5.12, 5.12)

    def evaluate(self, solution: np.ndarray) -> float:
        return np.sum(solution**2)

class RastriginFunction(ContinuousProblem):
    """
    Hàm Rastrigin: f(x) = 10n + sum(x_i^2 - 10cos(2pi*x_i))
    Global Minimum: 0 tại x = [0, 0, ..., 0]
    Rất nhiều cực trị địa phương (multimodal).
    """
    def __init__(self, dimensions: int = 10):
        super().__init__("Rastrigin Function", dimensions, -5.12, 5.12)

    def evaluate(self, solution: np.ndarray) -> float:
        return 10 * self.dimensions + np.sum(
            solution**2 - 10 * np.cos(2 * np.pi * solution)
        )