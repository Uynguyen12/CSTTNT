import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==========================================================
# 1. GRID SEARCH ANIMATION
# ==========================================================

def animate_grid_search(grid,
                        visited_order,
                        frontier_history=None,
                        path=None,
                        interval=40,
                        save_path=None,
                        title="Grid Search Animation"):

    if not visited_order:
        print("No visited nodes to animate.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    display_map = np.zeros((*grid.shape, 3))

    def reset_base():
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == 1:
                    display_map[r, c] = [0, 0, 0]
                else:
                    display_map[r, c] = [1, 1, 1]

    reset_base()

    img = ax.imshow(display_map)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    total_frames = len(visited_order) + 20

    def update(frame):

        reset_base()

        # visited
        for i in range(min(frame, len(visited_order))):
            r, c = visited_order[i]
            display_map[r, c] = [0.7, 0.7, 0.7]

        # path
        if frame >= len(visited_order) and path:
            for r, c in path:
                display_map[r, c] = [1, 0, 0]

        img.set_data(display_map)
        return [img]

    anim = FuncAnimation(fig, update,
                         frames=total_frames,
                         interval=interval,
                         repeat=False)

    if save_path:
        anim.save(save_path, writer="pillow", fps=1000 // interval)

    plt.show()


# ==========================================================
# 2. TSP ANIMATION
# ==========================================================

def animate_tsp(cities,
                best_route_history,
                interval=100,
                save_path=None,
                title="TSP Optimization Animation"):

    if not best_route_history:
        print("No route history to animate.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(cities[:, 0], cities[:, 1])
    line, = ax.plot([], [], 'r-')

    total_frames = len(best_route_history)

    def update(frame):

        route = np.asarray(best_route_history[frame])

        # bảo vệ chống float hoặc lỗi
        if not np.issubdtype(route.dtype, np.integer):
            return []

        coords = cities[route]
        coords = np.vstack([coords, coords[0]])

        line.set_data(coords[:, 0], coords[:, 1])
        ax.set_title(f"{title} - Iteration {frame}")

        return [line]

    anim = FuncAnimation(fig, update,
                         frames=total_frames,
                         interval=interval,
                         repeat=False)

    if save_path:
        anim.save(save_path, writer="pillow", fps=1000 // interval)

    plt.show()


# ==========================================================
# 3. CONTINUOUS OPTIMIZATION ANIMATION (2D)
# ==========================================================

def animate_continuous(objective_func,
                       bounds,
                       best_history=None,
                       population_history=None,
                       resolution=200,
                       interval=80,
                       save_path=None,
                       title="Continuous Optimization"):

    if not best_history and not population_history:
        print("No history to animate.")
        return

    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = objective_func(np.array([X[i, j], Y[i, j]]))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.contourf(X, Y, Z, levels=50)

    best_dot, = ax.plot([], [], 'ro')
    pop_scatter = ax.scatter([], [])

    total_frames = 0

    if best_history:
        total_frames = len(best_history)

    if population_history:
        total_frames = max(total_frames, len(population_history))

    def update(frame):

        if population_history and frame < len(population_history):
            pop_scatter.set_offsets(population_history[frame])

        if best_history and frame < len(best_history):
            bx, by = best_history[frame]
            best_dot.set_data(bx, by)

        ax.set_title(f"{title} - Iteration {frame}")
        return [best_dot, pop_scatter]

    anim = FuncAnimation(fig, update,
                         frames=total_frames,
                         interval=interval,
                         repeat=False)

    if save_path:
        anim.save(save_path, writer="pillow", fps=1000 // interval)

    plt.show()
