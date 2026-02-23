"""
Comprehensive Benchmark for Swarm-Based Algorithms

Chạy tất cả thuật toán swarm-based trên:
1. Continuous optimization problems (Sphere, Rastrigin, etc.)
2. Discrete optimization problems (TSP, Knapsack, Graph Coloring)

Với analysis đầy đủ:
- Convergence curves
- Performance comparison
- Parameter sensitivity
- Statistical analysis
- Scalability analysis
"""

import sys
from pathlib import Path
import numpy as np
import time
from typing import Dict, List, Tuple
import pickle

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import algorithms
from algorithms.nature_inspired.swarm_based import (
    ParticleSwarmOptimization,
    AntColonyOptimization,
    ArtificialBeeColony,
    FireflyAlgorithm,
    CuckooSearch
)

# Import test cases and visualization
from test_cases_continuous import get_all_benchmark_functions
from visualization_swarm import SwarmVisualizer, save_results


class SwarmBenchmark:
    """
    Comprehensive benchmark for swarm algorithms.
    """
    
    def __init__(self, num_runs: int = 30, save_dir: str = 'benchmark_results'):
        """
        Args:
            num_runs: Number of independent runs per experiment
            save_dir: Directory to save results
        """
        self.num_runs = num_runs
        self.save_dir = save_dir
        self.visualizer = SwarmVisualizer(save_dir=f'{save_dir}/visualizations')
        
        import os
        os.makedirs(save_dir, exist_ok=True)
    
    def create_algorithm_configs(self) -> Dict:
        """
        Create algorithm configurations for CONTINUOUS problems.
        
        Note: ACO is excluded as it's designed for discrete problems.
        
        Returns:
            Dictionary of algorithm configurations
        """
        return {
            'PSO': {
                'class': ParticleSwarmOptimization,
                'params': {
                    'population_size': 50,
                    'max_iterations': 200,
                    'inertia_weight': 0.7,
                    'cognitive_coeff': 1.5,
                    'social_coeff': 1.5,
                }
            },
            'ABC': {
                'class': ArtificialBeeColony,
                'params': {
                    'population_size': 50,
                    'max_iterations': 200,
                    'limit': None,  # Will be set automatically
                }
            },
            'FA': {
                'class': FireflyAlgorithm,
                'params': {
                    'population_size': 50,
                    'max_iterations': 200,
                    'alpha': 0.5,
                    'beta_min': 0.2,
                    'gamma': 1.0,
                }
            },
            'CS': {
                'class': CuckooSearch,
                'params': {
                    'population_size': 50,
                    'max_iterations': 200,
                    'pa': 0.25,
                    'alpha': 0.01,
                    'beta': 1.5,
                }
            }
        }
    
    def create_discrete_algorithm_configs(self) -> Dict:
        """
        Create algorithm configurations for DISCRETE problems.
        
        Returns:
            Dictionary of algorithm configurations for discrete optimization
        """
        return {
            'ACO': {
                'class': AntColonyOptimization,
                'params': {
                    'population_size': 50,
                    'max_iterations': 200,
                    'archive_size': 50,
                    'intensification_factor': 0.85,
                    'deviation_distance': 0.1,
                }
            }
        }
    
    def run_single_experiment(self,
                            algorithm_class,
                            algorithm_params: Dict,
                            benchmark_func,
                            seed: int) -> Dict:
        """
        Run single experiment.
        
        Returns:
            Dictionary with results
        """
        # Create problem dict (already includes bounds info)
        problem = benchmark_func.get_problem_dict()
        
        # Create algorithm (without bounds - they're in the problem dict)
        algorithm = algorithm_class(**algorithm_params, seed=seed, verbose=False)
        
        # Run optimization
        result = algorithm.optimize(problem)
        
        return {
            'fitness': result.fitness,
            'solution': result.solution,
            'convergence_curve': result.convergence_curve,
            'iterations': result.iterations,
            'evaluations': result.evaluations,
            'execution_time': result.execution_time,
            'success': result.success
        }
    
    def benchmark_continuous_problems(self) -> Dict:
        """
        Benchmark on continuous optimization problems.
        
        Returns:
            Dictionary with all results
        """
        print("=" * 80)
        print("BENCHMARKING CONTINUOUS OPTIMIZATION PROBLEMS")
        print("=" * 80)
        
        # Get benchmark functions
        all_functions = get_all_benchmark_functions()
        
        # Get algorithm configs
        algo_configs = self.create_algorithm_configs()
        
        # Results storage
        all_results = {
            'convergence': {},  # {func_name: {algo_name: [curves]}}
            'performance': {},  # {func_name: {algo_name: [fitness_values]}}
            'statistics': {},   # {func_name: {algo_name: {mean, std, best, worst}}}
        }
        
        # Test on selected functions
        test_functions = []
        
        # Select representative functions
        test_functions.extend(all_functions['unimodal'][:2])  # Sphere-5D, Sphere-10D
        test_functions.extend(all_functions['multimodal'][:4])  # Rastrigin, Ackley, etc.
        
        for func in test_functions:
            func_name = func.name
            print(f"\n{'='*80}")
            print(f"Function: {func_name}")
            print(f"Dimensions: {func.dimensions}")
            print(f"Bounds: {func.bounds[0]}")
            print(f"Global Minimum: {func.global_minimum}")
            print(f"{'='*80}")
            
            all_results['convergence'][func_name] = {}
            all_results['performance'][func_name] = {}
            all_results['statistics'][func_name] = {}
            
            for algo_name, algo_config in algo_configs.items():
                print(f"\n  Testing {algo_name}...")
                
                algo_class = algo_config['class']
                algo_params = algo_config['params'].copy()
                
                convergence_curves = []
                fitness_values = []
                
                # Run multiple times
                for run in range(self.num_runs):
                    if (run + 1) % 10 == 0:
                        print(f"    Run {run + 1}/{self.num_runs}...")
                    
                    result = self.run_single_experiment(
                        algo_class,
                        algo_params,
                        func,
                        seed=42 + run
                    )
                    
                    convergence_curves.append(result['convergence_curve'])
                    fitness_values.append(result['fitness'])
                
                # Store results
                all_results['convergence'][func_name][algo_name] = convergence_curves
                all_results['performance'][func_name][algo_name] = fitness_values
                
                # Calculate statistics
                all_results['statistics'][func_name][algo_name] = {
                    'mean': np.mean(fitness_values),
                    'std': np.std(fitness_values),
                    'best': np.min(fitness_values),
                    'worst': np.max(fitness_values),
                    'median': np.median(fitness_values)
                }
                
                # Print summary
                print(f"    Mean: {np.mean(fitness_values):.6f} ± {np.std(fitness_values):.6f}")
                print(f"    Best: {np.min(fitness_values):.6f}")
                print(f"    Worst: {np.max(fitness_values):.6f}")
        
        return all_results
    
    def benchmark_parameter_sensitivity(self) -> Dict:
        """
        Benchmark parameter sensitivity for PSO.
        
        Returns:
            Dictionary with sensitivity results
        """
        print("\n" + "=" * 80)
        print("PARAMETER SENSITIVITY ANALYSIS - PSO")
        print("=" * 80)
        
        from test_cases_continuous import SphereFunction
        func = SphereFunction(dimensions=10)
        
        sensitivity_results = {}
        
        # Test inertia weight
        print("\n1. Testing Inertia Weight...")
        inertia_values = [0.4, 0.6, 0.7, 0.8, 0.9]
        inertia_results = []
        
        for w in inertia_values:
            print(f"  w = {w}")
            fitness_values = []
            
            for run in range(20):  # 20 runs per parameter
                pso = ParticleSwarmOptimization(
                    population_size=50,
                    max_iterations=200,
                    inertia_weight=w,
                    cognitive_coeff=1.5,
                    social_coeff=1.5,
                    seed=42 + run,
                    verbose=False
                )
                
                result = pso.optimize(func.get_problem_dict())
                fitness_values.append(result.fitness)
            
            inertia_results.append(fitness_values)
            print(f"    Mean: {np.mean(fitness_values):.6f}")
        
        sensitivity_results['inertia_weight'] = {
            'parameter_values': inertia_values,
            'results': inertia_results
        }
        
        # Test population size
        print("\n2. Testing Population Size...")
        pop_sizes = [20, 30, 50, 75, 100]
        pop_results = []
        
        for pop_size in pop_sizes:
            print(f"  Population = {pop_size}")
            fitness_values = []
            
            for run in range(20):
                pso = ParticleSwarmOptimization(
                    population_size=pop_size,
                    max_iterations=200,
                    inertia_weight=0.7,
                    cognitive_coeff=1.5,
                    social_coeff=1.5,
                    seed=42 + run,
                    verbose=False
                )
                
                result = pso.optimize(func.get_problem_dict())
                fitness_values.append(result.fitness)
            
            pop_results.append(fitness_values)
            print(f"    Mean: {np.mean(fitness_values):.6f}")
        
        sensitivity_results['population_size'] = {
            'parameter_values': pop_sizes,
            'results': pop_results
        }
        
        return sensitivity_results
    
    def benchmark_scalability(self) -> Dict:
        """
        Test scalability with increasing dimensions.
        
        Returns:
            Dictionary with scalability results
        """
        print("\n" + "=" * 80)
        print("SCALABILITY ANALYSIS")
        print("=" * 80)
        
        from test_cases_continuous import SphereFunction, RastriginFunction
        
        dimensions_list = [5, 10, 20, 30, 50]
        algo_configs = self.create_algorithm_configs()
        
        scalability_results = {
            'Sphere': {},
            'Rastrigin': {}
        }
        
        for func_type in ['Sphere', 'Rastrigin']:
            print(f"\n{func_type} Function:")
            print("-" * 80)
            
            for algo_name, algo_config in algo_configs.items():
                print(f"\n  {algo_name}:")
                fitness_by_dim = []
                
                for dim in dimensions_list:
                    print(f"    Dimension {dim}...", end=' ')
                    
                    # Create function
                    if func_type == 'Sphere':
                        func = SphereFunction(dimensions=dim)
                    else:
                        func = RastriginFunction(dimensions=dim)
                    
                    # Run 10 times
                    fitness_values = []
                    for run in range(10):
                        algo_class = algo_config['class']
                        algo_params = algo_config['params'].copy()
                        # Don't add bounds to algo_params - it's in the problem dict
                        algo_params['max_iterations'] = 200
                        
                        algorithm = algo_class(**algo_params, seed=42+run, verbose=False)
                        result = algorithm.optimize(func.get_problem_dict())
                        fitness_values.append(result.fitness)
                    
                    mean_fitness = np.mean(fitness_values)
                    fitness_by_dim.append(mean_fitness)
                    print(f"Mean = {mean_fitness:.6f}")
                
                scalability_results[func_type][algo_name] = fitness_by_dim
        
        scalability_results['dimensions'] = dimensions_list
        
        return scalability_results
    
    def benchmark_discrete_problems(self) -> Dict:
        """
        Benchmark ACO on discrete optimization problems (TSP).
        
        Returns:
            Dictionary with discrete problem results
        """
        print("\n" + "=" * 80)
        print("BENCHMARKING DISCRETE OPTIMIZATION PROBLEMS (ACO)")
        print("=" * 80)
        
        try:
            # Try to load discrete test cases
            import pickle
            with open('discrete_test_cases.pkl', 'rb') as f:
                discrete_cases = pickle.load(f)
            
            print("\nLoaded discrete test cases from file")
            
            # Get ACO config
            aco_config = self.create_discrete_algorithm_configs()['ACO']
            
            discrete_results = {
                'tsp': {}
            }
            
            # Test on TSP instances
            for case_name, instance in list(discrete_cases.get('tsp', {}).items())[:2]:  # Test first 2
                print(f"\nTesting {case_name}...")
                
                fitness_values = []
                for run in range(min(10, self.num_runs)):  # Fewer runs for discrete
                    print(f"  Run {run + 1}...", end=' ')
                    
                    # Create ACO instance
                    aco_params = aco_config['params'].copy()
                    aco = aco_config['class'](**aco_params, seed=42+run, verbose=False)
                    
                    # For TSP, need special problem dict
                    problem = {
                        'objective_func': instance.evaluate_tour,
                        'num_cities': instance.num_cities,
                        'distance_matrix': instance.distance_matrix,
                        'is_discrete': True
                    }
                    
                    # Run optimization
                    result = aco.optimize(problem)
                    fitness_values.append(result.fitness)
                    print(f"Fitness: {result.fitness:.2f}")
                
                discrete_results['tsp'][case_name] = {
                    'mean': np.mean(fitness_values),
                    'std': np.std(fitness_values),
                    'best': np.min(fitness_values)
                }
                
                print(f"\n  Results for {case_name}:")
                print(f"    Mean: {np.mean(fitness_values):.2f}")
                print(f"    Best: {np.min(fitness_values):.2f}")
            
            return discrete_results
            
        except FileNotFoundError:
            print("\nWarning: discrete_test_cases.pkl not found")
            print("Skipping discrete problem benchmark")
            print("Run test_cases_discrete.py first to generate test cases")
            return {}
        except Exception as e:
            print(f"\nError in discrete benchmark: {e}")
            print("Skipping discrete problem benchmark")
            return {}
    
    def generate_visualizations(self, all_results: Dict):
        """
        Generate all visualizations.
        
        Args:
            all_results: Dictionary with all benchmark results
        """
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)
        
        # Convergence curves
        print("\n1. Convergence Curves...")
        for func_name, algo_results in all_results['convergence'].items():
            self.visualizer.plot_convergence_curves(
                algo_results,
                func_name,
                save_name=f'convergence_{func_name.replace(" ", "_")}'
            )
        
        # Performance comparison
        print("\n2. Performance Comparison...")
        self.visualizer.plot_performance_comparison(
            all_results['performance'],
            save_name='performance_comparison'
        )
        
        # Statistical analysis
        print("\n3. Statistical Analysis...")
        self.visualizer.plot_statistical_analysis(
            all_results['performance'],
            save_name='statistical_analysis'
        )
        
        # Parameter sensitivity
        if 'sensitivity' in all_results:
            print("\n4. Parameter Sensitivity...")
            for param_name, param_data in all_results['sensitivity'].items():
                self.visualizer.plot_parameter_sensitivity(
                    param_name,
                    param_data['parameter_values'],
                    param_data['results'],
                    'PSO',
                    'Sphere-10D',
                    save_name=f'sensitivity_{param_name}'
                )
        
        # Scalability
        if 'scalability' in all_results:
            print("\n5. Scalability Analysis...")
            for func_type in ['Sphere', 'Rastrigin']:
                self.visualizer.plot_scalability_analysis(
                    all_results['scalability']['dimensions'],
                    all_results['scalability'][func_type],
                    save_name=f'scalability_{func_type}'
                )
        
        print("\nAll visualizations saved!")
    
    def print_summary_table(self, all_results: Dict):
        """
        Print summary table of results.
        
        Args:
            all_results: Dictionary with all results
        """
        print("\n" + "=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)
        
        statistics = all_results['statistics']
        
        for func_name, algo_stats in statistics.items():
            print(f"\n{func_name}:")
            print("-" * 80)
            print(f"{'Algorithm':<10} {'Mean':<15} {'Std':<15} {'Best':<15} {'Worst':<15}")
            print("-" * 80)
            
            for algo_name, stats in algo_stats.items():
                print(f"{algo_name:<10} "
                      f"{stats['mean']:<15.6f} "
                      f"{stats['std']:<15.6f} "
                      f"{stats['best']:<15.6f} "
                      f"{stats['worst']:<15.6f}")
        
        # Ranking
        print("\n" + "=" * 80)
        print("ALGORITHM RANKING (by mean performance)")
        print("=" * 80)
        
        algo_names = list(next(iter(statistics.values())).keys())
        
        for func_name, algo_stats in statistics.items():
            print(f"\n{func_name}:")
            
            # Sort by mean
            sorted_algos = sorted(algo_stats.items(), key=lambda x: x[1]['mean'])
            
            for rank, (algo_name, stats) in enumerate(sorted_algos, 1):
                print(f"  {rank}. {algo_name:<10} Mean = {stats['mean']:.6f}")


def main():
    """
    Main benchmark execution.
    """
    print("=" * 80)
    print("COMPREHENSIVE SWARM ALGORITHM BENCHMARK")
    print("=" * 80)
    print("\nThis benchmark will:")
    print("  1. Test 4 swarm algorithms (PSO, ABC, FA, CS)")
    print("  2. On 6 benchmark functions")
    print("  3. With 30 independent runs each")
    print("  4. Parameter sensitivity analysis")
    print("  5. Scalability analysis")
    print("  6. Generate comprehensive visualizations")
    print("\nNote: ACO is designed for discrete problems and will be tested separately")
    print("\nEstimated time: 15-30 minutes")
    
    input("\nPress Enter to start...")
    
    # Create benchmark
    benchmark = SwarmBenchmark(num_runs=30, save_dir='swarm_benchmark_results')
    
    # Run benchmarks
    print("\n" + "=" * 80)
    print("PHASE 1: CONTINUOUS OPTIMIZATION BENCHMARK")
    print("=" * 80)
    results_continuous = benchmark.benchmark_continuous_problems()
    
    print("\n" + "=" * 80)
    print("PHASE 2: PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 80)
    results_sensitivity = benchmark.benchmark_parameter_sensitivity()
    
    print("\n" + "=" * 80)
    print("PHASE 3: SCALABILITY ANALYSIS")
    print("=" * 80)
    results_scalability = benchmark.benchmark_scalability()
    
    # Combine all results
    all_results = {
        'convergence': results_continuous['convergence'],
        'performance': results_continuous['performance'],
        'statistics': results_continuous['statistics'],
        'sensitivity': results_sensitivity,
        'scalability': results_scalability
    }
    
    # Save results
    save_results(all_results, 'swarm_benchmark_results/all_results.pkl')
    
    # Generate visualizations
    benchmark.generate_visualizations(all_results)
    
    # Print summary
    benchmark.print_summary_table(all_results)
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: swarm_benchmark_results/")
    print(f"Visualizations saved to: swarm_benchmark_results/visualizations/")


if __name__ == "__main__":
    main()