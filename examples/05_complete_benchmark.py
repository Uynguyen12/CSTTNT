"""
Example 5: Complete Benchmark - All Algorithms Comparison
Chạy tất cả thuật toán trên một bộ test functions để so sánh hiệu năng
"""

import sys
from pathlib import Path
import numpy as np
import time
from typing import Callable, Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Classical Search
from utils.graph import Graph
from algorithms.classical.uninformed import BreadthFirstSearch, DepthFirstSearch
from algorithms.classical.informed import AStarSearch

# Evolution-Based
from algorithms.nature_inspired.evolution_based import GeneticAlgorithm, DifferentialEvolution

# Swarm-Based
from algorithms.nature_inspired.swarm_based import ParticleSwarmOptimization

# Physics/Human-Based
from algorithms.nature_inspired.physics_based import SimulatedAnnealing, HillClimbing, HarmonySearch


# ============= Test Functions =============
class OptimizationBenchmark:
    """Collection of benchmark functions for optimization"""
    
    @staticmethod
    def sphere(x):
        """Sphere: Simple, unimodal"""
        return np.sum(x**2)
    
    @staticmethod
    def rastrigin(x):
        """Rastrigin: Highly multimodal"""
        return 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))
    
    @staticmethod
    def schwefel(x):
        """Schwefel: Highly deceptive"""
        n = len(x)
        return 418.9829*n - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    
    @staticmethod
    def ackley(x):
        """Ackley: Multiple local minima"""
        n = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(2*np.pi*x))
        return -20*np.exp(-0.2*np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + np.e


class AlgorithmBenchmark:
    """Benchmark suite for comparing all algorithms"""
    
    def __init__(self):
        self.results: Dict[str, Dict] = {}
    
    def benchmark_continuous_optimizers(self):
        """Benchmark continuous optimization algorithms"""
        print("=" * 80)
        print("CONTINUOUS OPTIMIZATION BENCHMARK")
        print("=" * 80)
        
        dimensions = 3
        bounds = [(-5.0, 5.0)] * dimensions
        
        functions = {
            "Sphere": OptimizationBenchmark.sphere,
            "Rastrigin": OptimizationBenchmark.rastrigin,
            "Schwefel": OptimizationBenchmark.schwefel,
            "Ackley": OptimizationBenchmark.ackley,
        }
        
        algorithms = {
            "GA": GeneticAlgorithm(
                population_size=50,
                max_iterations=200,
                bounds=bounds,
                seed=42
            ),
            "DE": DifferentialEvolution(
                population_size=50,
                max_iterations=200,
                bounds=bounds,
                seed=42
            ),
            "PSO": ParticleSwarmOptimization(
                population_size=50,
                max_iterations=200,
                bounds=bounds,
                seed=42
            ),
            "SA": SimulatedAnnealing(
                bounds=bounds,
                max_iterations=10000,
                initial_temperature=100.0,
                cooling_rate=0.995,
                seed=42
            ),
            "HC": HillClimbing(
                bounds=bounds,
                max_iterations=10000,
                step_size=0.5,
                seed=42
            ),
            "HS": HarmonySearch(
                harmony_memory_size=30,
                max_iterations=200,
                bounds=bounds,
                seed=42
            ),
        }
        
        print(f"\nDimensions: {dimensions}")
        print(f"Bounds: {bounds[0]}")
        print(f"Number of test functions: {len(functions)}")
        print(f"Number of algorithms: {len(algorithms)}\n")
        
        # Run benchmarks
        for func_name, func in functions.items():
            print("\n" + "=" * 80)
            print(f"FUNCTION: {func_name}")
            print("=" * 80)
            print("{:<15} {:<20} {:<15} {:<15}".format(
                "Algorithm", "Best Fitness", "Iterations", "Time (s)"
            ))
            print("-" * 80)
            
            self.results[func_name] = {}
            
            for algo_name, algo in algorithms.items():
                try:
                    problem = {
                        'objective_func': func,
                        'dimensions': dimensions,
                        'lower_bound': bounds[0][0],
                        'upper_bound': bounds[0][1],
                        'is_discrete': False
                    }
                    result = algo.optimize(problem)
                    self.results[func_name][algo_name] = result
                    
                    print("{:<15} {:<20.8f} {:<15} {:<15.4f}".format(
                        algo_name,
                        result.fitness,
                        result.iterations,
                        result.execution_time
                    ))
                except Exception as e:
                    print("{:<15} {:<20} {:<15}".format(algo_name, "ERROR", str(e)[:15]))
    
    def benchmark_graph_search(self):
        """Benchmark graph search algorithms"""
        print("\n" + "=" * 80)
        print("GRAPH SEARCH BENCHMARK")
        print("=" * 80)
        
        # Create sample graph
        graph = Graph(num_nodes=10)
        
        # Create a grid-like structure
        for i in range(10):
            if i < 9:
                graph.add_edge(i, i+1, 1.0)
            if i % 3 < 2 and i + 3 < 10:
                graph.add_edge(i, i+3, 1.5)
        
        # Add some extra connections
        graph.add_edge(0, 5, 2.0)
        graph.add_edge(2, 7, 1.0)
        
        start = 0
        goal = 9
        
        def heuristic(node, goal):
            """Simple Manhattan-like heuristic"""
            return abs(goal - node) * 0.8
        
        algorithms = {
            "BFS": BreadthFirstSearch(),
            "DFS": DepthFirstSearch(),
            "A*": AStarSearch(heuristic=heuristic),
        }
        
        print(f"\nGraph: {graph.num_nodes} nodes")
        print(f"Start: Node {start}, Goal: Node {goal}")
        print("\n{:<15} {:<15} {:<20} {:<15}".format(
            "Algorithm", "Path Found", "Cost", "Nodes Explored"
        ))
        print("-" * 80)
        
        for algo_name, algo in algorithms.items():
            try:
                result = algo.search(graph, start, goal)
                status = "✓" if result.path_found else "✗"
                print("{:<15} {:<15} {:<20.2f} {:<15}".format(
                    algo_name,
                    status,
                    result.cost,
                    result.nodes_explored
                ))
            except Exception as e:
                print("{:<15} {:<15} {:<20}".format(algo_name, "ERROR", str(e)[:20]))
    
    def print_summary(self):
        """Print comprehensive summary"""
        print("\n" + "=" * 80)
        print("OPTIMIZATION RESULTS SUMMARY")
        print("=" * 80)
        
        print("\nBest Algorithm per Function (by final fitness):")
        print("-" * 80)
        print("{:<20} {:<15} {:<15}".format("Function", "Best Algorithm", "Fitness"))
        print("-" * 80)
        
        for func_name, results in self.results.items():
            if results:
                best_algo = min(results.items(), key=lambda x: x[1].fitness)
                print("{:<20} {:<15} {:<15.8f}".format(
                    func_name,
                    best_algo[0],
                    best_algo[1].fitness
                ))
        
        # Overall statistics
        print("\n" + "=" * 80)
        print("OVERALL STATISTICS")
        print("=" * 80)
        
        all_results = {}
        for func_results in self.results.values():
            for algo_name, result in func_results.items():
                if algo_name not in all_results:
                    all_results[algo_name] = []
                all_results[algo_name].append(result.fitness)
        
        print("\n{:<15} {:<20} {:<20}".format(
            "Algorithm", "Average Fitness", "Wins"
        ))
        print("-" * 80)
        
        for algo_name, fitnesses in sorted(all_results.items()):
            avg_fitness = np.mean(fitnesses)
            
            # Count wins
            wins = 0
            for func_results in self.results.values():
                if algo_name in func_results:
                    best_fitness = min(r.fitness for r in func_results.values())
                    if func_results[algo_name].fitness == best_fitness:
                        wins += 1
            
            print("{:<15} {:<20.8f} {:<20}".format(
                algo_name,
                avg_fitness,
                wins
            ))


def main():
    """Run complete benchmark"""
    benchmark = AlgorithmBenchmark()
    
    # Run benchmarks
    benchmark.benchmark_continuous_optimizers()
    benchmark.benchmark_graph_search()
    
    # Print summary
    benchmark.print_summary()
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print("\nNotes:")
    print("- Each algorithm was run with the same random seed for reproducibility")
    print("- Different algorithms have different characteristics:")
    print("  * GA: Good balance of exploration and exploitation")
    print("  * DE: Fast convergence, especially for smooth functions")
    print("  * PSO: Good for multimodal problems, intuitive")
    print("  * SA: Can escape local optima via temperature control")
    print("  * HC: Fast but prone to local optima")
    print("  * HS: Inspired by music harmony, good for discrete/continuous problems")


if __name__ == "__main__":
    main()
