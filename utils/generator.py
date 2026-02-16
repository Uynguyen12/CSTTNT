import numpy as np


def generate_grid_map(rows=20,
                      cols=20,
                      obstacle_prob=0.2,
                      seed=None):
    """
    Generate random grid map.

    0 = free cell
    1 = obstacle

    Args:
        rows: số hàng
        cols: số cột
        obstacle_prob: xác suất obstacle
        seed: random seed để reproducible

    Returns:
        grid, start_id, goal_id
    """

    if seed is not None:
        np.random.seed(seed)

    grid = np.random.choice(
        [0, 1],
        size=(rows, cols),
        p=[1 - obstacle_prob, obstacle_prob]
    )

    # đảm bảo start và goal không bị block
    grid[0, 0] = 0
    grid[rows - 1, cols - 1] = 0

    start = 0
    goal = rows * cols - 1

    return grid, start, goal
