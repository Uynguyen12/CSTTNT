import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from utils.visualization.animation import (
    animate_grid_search,
    animate_tsp,
    animate_continuous
)

# --- IMPORT PROBLEMS ---
from problems.continous import SphereFunction, RastriginFunction
from problems.discrete import TSP

# --- IMPORT ALGORITHMS ---
from algorithms.nature_inspired.swarm_based import (
    ParticleSwarmOptimization,
    AntColonyOptimization,
    ArtificialBeeColony,
    FireflyAlgorithm
)
from algorithms.nature_inspired.evolution_based import (
    GeneticAlgorithm,
    DifferentialEvolution
)
from algorithms.nature_inspired.physics_based import (
    SimulatedAnnealing,
    HillClimbing
)
from algorithms.classical.uninformed import (
    BreadthFirstSearch,
    DepthFirstSearch,
    UniformCostSearch
)
from algorithms.classical.informed import AStarSearch

from utils.generator import generate_grid_map
from utils.graph import Graph
from utils.visualization.graph_plot import plot_grid_path
from utils.visualization.discrete_plot import plot_tsp_route


# ============================================================
# HELPER: RUN MULTIPLE TIMES
# ============================================================

def run_multiple(algo_factory, problem_dict, runs=20):

    fitness_list = []
    histories = []
    times = []

    for _ in range(runs):
        algo = algo_factory()
        start = time.perf_counter()
        res = algo.optimize(problem_dict)
        end = time.perf_counter()

        fitness_list.append(res.fitness)
        histories.append(res.convergence_curve)
        times.append(end - start)

    min_len = min(len(h) for h in histories)
    trimmed = [h[:min_len] for h in histories]
    mean_curve = np.mean(trimmed, axis=0)

    return {
        "fitness_all": fitness_list,
        "mean_curve": mean_curve,
        "mean_fitness": np.mean(fitness_list),
        "std_fitness": np.std(fitness_list),
        "best": np.min(fitness_list),
        "success_rate": np.mean(np.array(fitness_list) < 1e-3),
        "mean_time": np.mean(times)
    }


# ============================================================
# EXPERIMENT 1 – SPHERE
# ============================================================

def experiment_sphere():

    print("\n===== EXPERIMENT 1: SPHERE =====")

    dimensions_list = [10, 30, 50]
    runs = 20

    for dim in dimensions_list:

        print(f"\n--- Dimension = {dim} ---")
        problem = SphereFunction(dimensions=dim)
        p_dict = problem.to_dict()

        algorithms = {
            "PSO": lambda: ParticleSwarmOptimization(population_size=30, max_iterations=150),
            "DE": lambda: DifferentialEvolution(population_size=30, max_iterations=150),
            "GA": lambda: GeneticAlgorithm(population_size=30, max_iterations=150),
            "ABC": lambda: ArtificialBeeColony(population_size=30, max_iterations=150),
            "FA": lambda: FireflyAlgorithm(population_size=30, max_iterations=150)
        }

        results = {}

        for name, factory in algorithms.items():
            print(f"Running {name} ...")
            stats = run_multiple(factory, p_dict, runs=runs)
            results[name] = stats
            print(f"Mean Fitness: {stats['mean_fitness']:.2e} ± {stats['std_fitness']:.2e}")

        # Plot convergence
        plt.figure(figsize=(8, 5))
        for name in algorithms.keys():
            plt.plot(results[name]["mean_curve"], label=name)

        plt.title(f"Sphere (Dim={dim}) - Mean Convergence")
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.legend()
        plt.yscale("log")
        plt.show()

        # Animation chỉ khi 2D
        if dim == 2:
            for name, factory in algorithms.items():
                algo = factory()
                algo.enable_animation = True
                res = algo.optimize(p_dict)

                animate_continuous(
                    objective_func=p_dict["objective_func"],
                    bounds=p_dict["bounds"],
                    best_history=algo.best_route_history,
                    population_history=algo.position_history,
                    interval=80,
                    title=f"{name} - Sphere 2D"
                )

# ============================================================
# EXPERIMENT 2 – RASTRIGIN
# ============================================================

def experiment_rastrigin():

    print("\n===== EXPERIMENT 2: RASTRIGIN =====")

    problem = RastriginFunction(dimensions=20)
    p_dict = problem.to_dict()
    runs = 25

    algorithms = {
        "GA": lambda: GeneticAlgorithm(population_size=50, max_iterations=200),
        "PSO": lambda: ParticleSwarmOptimization(population_size=50, max_iterations=200),
        "ABC": lambda: ArtificialBeeColony(population_size=50, max_iterations=200),
        "SA": lambda: SimulatedAnnealing(max_iterations=10000)
    }

    table = []
    all_results = {}

    for name, factory in algorithms.items():
        print(f"Running {name} ...")
        stats = run_multiple(factory, p_dict, runs=runs)
        all_results[name] = stats

        table.append({
            "Algorithm": name,
            "Mean": stats["mean_fitness"],
            "Std": stats["std_fitness"],
            "Best": stats["best"],
            "Success Rate": stats["success_rate"]
        })

    df = pd.DataFrame(table)
    print(df)

    plt.figure(figsize=(8, 5))
    plt.boxplot(
        [all_results[name]["fitness_all"] for name in algorithms.keys()],
        tick_labels=list(algorithms.keys())
    )

    plt.title("Rastrigin Stability (Lower is Better)")
    plt.ylabel("Fitness")
    plt.show()


# ============================================================
# EXPERIMENT 3 – TSP
# ============================================================

def tsp_wrapper(x, eval_func):
    perm = np.argsort(x)
    return eval_func(perm)


def experiment_tsp():

    print("\n===== EXPERIMENT 3: TSP =====")

    cities_list = [20, 40]
    runs = 15

    for n in cities_list:

        print(f"\n--- Cities = {n} ---")
        tsp = TSP(num_cities=n, seed=42)

        p_dict = tsp.to_dict()
        p_dict["objective_func"] = tsp.evaluate  # không dùng argsort wrapper nữa

        # Chỉ giữ thuật toán phù hợp TSP thật sự
        algorithms = {
            "ACO": lambda: AntColonyOptimization(population_size=60, max_iterations=150),
            "GA": lambda: GeneticAlgorithm(population_size=60, max_iterations=150)
        }

        table = []

        for name, factory in algorithms.items():
            stats = run_multiple(factory, p_dict, runs=runs)
            table.append({
                "Algorithm": name,
                "Mean Cost": stats["mean_fitness"],
                "Std": stats["std_fitness"],
                "Best": stats["best"]
            })

        df = pd.DataFrame(table)
        print(df)

        # Animate best algorithm
        best_algo = min(table, key=lambda x: x["Best"])["Algorithm"]
        best_factory = algorithms[best_algo]

        algo = best_factory()
        algo.enable_animation = True
        res = algo.optimize(p_dict)

        # route lấy trực tiếp từ history
        best_route = np.array(algo.best_route_history[-1], dtype=int)

        plot_tsp_route(
            tsp.city_coords,
            best_route,
            title=f"TSP Best Route - {best_algo}"
        )

        animate_tsp(
            tsp.city_coords,
            best_route_history=[
                np.array(r, dtype=int) for r in algo.best_route_history
            ],
            interval=100,
            title=f"TSP Optimization - {best_algo}"
        )


# ============================================================
# EXPERIMENT 4 – GRID MAP
# ============================================================

def experiment_grid():

    print("\n===== EXPERIMENT 4: GRID MAP =====")

    sizes = [20, 30, 50]

    for size in sizes:

        print(f"\n--- Grid {size}x{size} ---")

        grid, start, goal = generate_grid_map(
            size, size,
            obstacle_prob=0.2,
            seed=42
        )

        graph = Graph(grid)

        algorithms = {
            "BFS": BreadthFirstSearch(),
            "DFS": DepthFirstSearch(),
            "UCS": UniformCostSearch(),
            "A*": AStarSearch()
        }

        table = []

        for name, algo in algorithms.items():

            start_time = time.perf_counter()
            res = algo.search(graph, start, goal)
            end_time = time.perf_counter()

            if res.path_found:

                path_coords = [graph.id_to_coord(n) for n in res.path]
                visited_coords = [graph.id_to_coord(
                    n) for n in res.explored_order]

                animate_grid_search(
                    graph.grid,
                    visited_order=visited_coords,
                    path=path_coords,
                    interval=30,
                    title=f"{name} - Grid {size}x{size}"
                )

            table.append({
                "Algorithm": name,
                "Cost": res.cost if res.path_found else None,
                "Nodes Explored": res.nodes_explored,
                "Time (ms)": (end_time - start_time) * 1000
            })

        df = pd.DataFrame(table)
        print(df)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    experiment_sphere()
    experiment_rastrigin()
    experiment_tsp()
    experiment_grid()
