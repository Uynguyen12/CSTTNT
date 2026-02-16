import matplotlib.pyplot as plt
import numpy as np

def plot_grid_path(grid, path=None, visited_nodes=None, title="Grid Map Search"):

    plt.figure(figsize=(8, 8))

    # Tạo map hiển thị riêng, không sửa grid gốc
    display_map = np.zeros((*grid.shape, 3))  # RGB image

    # 1. Vẽ free (trắng) và obstacle (đen)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 1:       # obstacle
                display_map[r, c] = [0, 0, 0]
            else:                     # free
                display_map[r, c] = [1, 1, 1]

    # 2. Vẽ visited nodes (xám nhạt)
    if visited_nodes is not None:
        for r, c in visited_nodes:
            display_map[r, c] = [0.7, 0.7, 0.7]

    # 3. Vẽ path (đỏ đậm)
    if path is not None:
        for r, c in path:
            display_map[r, c] = [1, 0, 0]

        # Start (xanh lá)
        sr, sc = path[0]
        display_map[sr, sc] = [0, 1, 0]

        # End (xanh dương)
        er, ec = path[-1]
        display_map[er, ec] = [0, 0, 1]

    plt.imshow(display_map)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.show()
