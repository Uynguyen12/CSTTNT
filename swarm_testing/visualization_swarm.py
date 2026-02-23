"""
Comprehensive Visualization for Swarm-Based Algorithms

Visualizations:
1. Convergence curves
2. Performance comparison (box plots, bar charts)
3. Parameter sensitivity analysis
4. 3D surface plots with optimization trajectory
5. Population/swarm behavior over time
6. Statistical analysis (mean, std, best/worst runs)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Tuple, Optional
import pickle


# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class SwarmVisualizer:
    """
    Comprehensive visualization tool for swarm algorithms.
    """
    
    def __init__(self, save_dir: str = 'visualizations'):
        """
        Args:
            save_dir: Directory to save figures
        """
        self.save_dir = save_dir
        import os
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_convergence_curves(self,
                                results_dict: Dict[str, List],
                                function_name: str,
                                save_name: Optional[str] = None):
        """
        Plot convergence curves for multiple algorithms.
        
        Args:
            results_dict: {algorithm_name: [convergence_curve_run1, ...]}
            function_name: Name of benchmark function
            save_name: Filename to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
        
        for (algo_name, convergence_curves), color in zip(results_dict.items(), colors):
            # Calculate statistics across multiple runs
            min_length = min(len(curve) for curve in convergence_curves)
            curves_array = np.array([curve[:min_length] for curve in convergence_curves])
            
            mean_curve = np.mean(curves_array, axis=0)
            std_curve = np.std(curves_array, axis=0)
            best_curve = np.min(curves_array, axis=0)
            
            iterations = np.arange(len(mean_curve))
            
            # Plot 1: Mean with std deviation band
            ax1.plot(iterations, mean_curve, label=algo_name, color=color, linewidth=2)
            ax1.fill_between(iterations,
                            mean_curve - std_curve,
                            mean_curve + std_curve,
                            alpha=0.2,
                            color=color)
            
            # Plot 2: Best run (log scale)
            ax2.semilogy(iterations, best_curve, label=algo_name, color=color, linewidth=2)
        
        # Formatting
        ax1.set_xlabel('Iterations', fontsize=12)
        ax1.set_ylabel('Fitness Value', fontsize=12)
        ax1.set_title(f'Convergence: {function_name}\n(Mean ± Std)', fontsize=14)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Iterations', fontsize=12)
        ax2.set_ylabel('Fitness Value (log scale)', fontsize=12)
        ax2.set_title(f'Best Run: {function_name}', fontsize=14)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f'{self.save_dir}/{save_name}.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{self.save_dir}/{save_name}.pdf', bbox_inches='tight')
        
        plt.show()
    
    def plot_performance_comparison(self,
                                   results_dict: Dict[str, Dict[str, List[float]]],
                                   save_name: Optional[str] = None):
        """
        Compare algorithm performance across multiple functions.
        
        Args:
            results_dict: {function_name: {algorithm_name: [fitness_values]}}
            save_name: Filename to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        function_names = list(results_dict.keys())
        algorithm_names = list(next(iter(results_dict.values())).keys())
        
        # Plot 1: Box plots for each function
        for idx, func_name in enumerate(function_names[:4]):  # Max 4 functions
            if idx >= 4:
                break
            
            data = []
            labels = []
            for algo_name in algorithm_names:
                data.append(results_dict[func_name][algo_name])
                labels.append(algo_name)
            
            bp = axes[idx].boxplot(data, labels=labels, patch_artist=True)
            
            # Color boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            axes[idx].set_title(f'{func_name}', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Fitness Value', fontsize=10)
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f'{self.save_dir}/{save_name}_boxplot.png', dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Plot 2: Bar chart with error bars
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(function_names))
        width = 0.8 / len(algorithm_names)
        
        for i, algo_name in enumerate(algorithm_names):
            means = [np.mean(results_dict[func][algo_name]) for func in function_names]
            stds = [np.std(results_dict[func][algo_name]) for func in function_names]
            
            ax.bar(x + i * width, means, width, label=algo_name,
                  yerr=stds, capsize=5, alpha=0.8)
        
        ax.set_xlabel('Benchmark Function', fontsize=12)
        ax.set_ylabel('Mean Fitness Value', fontsize=12)
        ax.set_title('Algorithm Performance Comparison (Mean ± Std)', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(algorithm_names) - 1) / 2)
        ax.set_xticklabels(function_names, rotation=45, ha='right')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f'{self.save_dir}/{save_name}_barchart.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_parameter_sensitivity(self,
                                  parameter_name: str,
                                  parameter_values: List,
                                  results: List[List[float]],
                                  algorithm_name: str,
                                  function_name: str,
                                  save_name: Optional[str] = None):
        """
        Plot parameter sensitivity analysis.
        
        Args:
            parameter_name: Name of parameter being varied
            parameter_values: List of parameter values tested
            results: List of fitness value lists for each parameter
            algorithm_name: Name of algorithm
            function_name: Name of benchmark function
            save_name: Filename to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Calculate statistics
        means = [np.mean(r) for r in results]
        stds = [np.std(r) for r in results]
        mins = [np.min(r) for r in results]
        maxs = [np.max(r) for r in results]
        
        # Plot 1: Mean with error bars
        ax1.errorbar(parameter_values, means, yerr=stds,
                    fmt='o-', linewidth=2, markersize=8,
                    capsize=5, capthick=2, label='Mean ± Std')
        ax1.set_xlabel(parameter_name, fontsize=12)
        ax1.set_ylabel('Fitness Value', fontsize=12)
        ax1.set_title(f'Parameter Sensitivity: {algorithm_name}\n{function_name}',
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Box plots for each parameter value
        ax2.boxplot(results, labels=[str(v) for v in parameter_values],
                   patch_artist=True)
        ax2.set_xlabel(parameter_name, fontsize=12)
        ax2.set_ylabel('Fitness Value', fontsize=12)
        ax2.set_title('Distribution of Results', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f'{self.save_dir}/{save_name}.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_3d_surface_with_trajectory(self,
                                       func: callable,
                                       bounds: Tuple[float, float],
                                       best_positions: List[np.ndarray],
                                       algorithm_name: str,
                                       save_name: Optional[str] = None):
        """
        Plot 3D surface of 2D function with optimization trajectory.
        
        Args:
            func: 2D benchmark function
            bounds: (min, max) for both dimensions
            best_positions: List of best positions over iterations
            algorithm_name: Name of algorithm
            save_name: Filename to save plot
        """
        # Create meshgrid
        x = np.linspace(bounds[0], bounds[1], 100)
        y = np.linspace(bounds[0], bounds[1], 100)
        X, Y = np.meshgrid(x, y)
        
        # Evaluate function
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = func(np.array([X[i, j], Y[i, j]]))
        
        # Create figure
        fig = plt.figure(figsize=(16, 6))
        
        # Plot 1: 3D Surface with trajectory
        ax1 = fig.add_subplot(121, projection='3d')
        
        surf = ax1.plot_surface(X, Y, Z, cmap='viridis',
                               alpha=0.6, edgecolor='none')
        
        # Plot trajectory
        if len(best_positions) > 0:
            positions = np.array(best_positions)
            fitness_values = [func(p) for p in positions]
            
            ax1.plot(positions[:, 0], positions[:, 1], fitness_values,
                    'r-', linewidth=2, label='Optimization Path')
            ax1.scatter(positions[0, 0], positions[0, 1], fitness_values[0],
                       c='green', s=100, marker='o', label='Start')
            ax1.scatter(positions[-1, 0], positions[-1, 1], fitness_values[-1],
                       c='red', s=100, marker='*', label='End')
        
        ax1.set_xlabel('X', fontsize=10)
        ax1.set_ylabel('Y', fontsize=10)
        ax1.set_zlabel('f(x, y)', fontsize=10)
        ax1.set_title(f'{algorithm_name} - 3D Trajectory', fontsize=12, fontweight='bold')
        ax1.legend()
        fig.colorbar(surf, ax=ax1, shrink=0.5)
        
        # Plot 2: 2D Contour with trajectory
        ax2 = fig.add_subplot(122)
        
        contour = ax2.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.8)
        ax2.contour(X, Y, Z, levels=15, colors='black', alpha=0.3, linewidths=0.5)
        
        if len(best_positions) > 0:
            positions = np.array(best_positions)
            ax2.plot(positions[:, 0], positions[:, 1],
                    'r-', linewidth=2, label='Path')
            ax2.scatter(positions[0, 0], positions[0, 1],
                       c='green', s=100, marker='o', zorder=5, label='Start')
            ax2.scatter(positions[-1, 0], positions[-1, 1],
                       c='red', s=150, marker='*', zorder=5, label='End')
        
        ax2.set_xlabel('X', fontsize=12)
        ax2.set_ylabel('Y', fontsize=12)
        ax2.set_title(f'{algorithm_name} - Contour Plot', fontsize=12, fontweight='bold')
        ax2.legend()
        fig.colorbar(contour, ax=ax2)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f'{self.save_dir}/{save_name}.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_statistical_analysis(self,
                                 results_dict: Dict[str, Dict[str, List[float]]],
                                 save_name: Optional[str] = None):
        """
        Statistical analysis: mean, median, std, best/worst.
        
        Args:
            results_dict: {function_name: {algorithm_name: [fitness_values]}}
            save_name: Filename to save plot
        """
        function_names = list(results_dict.keys())
        algorithm_names = list(next(iter(results_dict.values())).keys())
        
        # Prepare data
        metrics = ['Mean', 'Median', 'Std', 'Best', 'Worst']
        n_metrics = len(metrics)
        n_algos = len(algorithm_names)
        
        fig, axes = plt.subplots(len(function_names), n_metrics,
                                figsize=(20, 4 * len(function_names)))
        
        if len(function_names) == 1:
            axes = axes.reshape(1, -1)
        
        for func_idx, func_name in enumerate(function_names):
            for metric_idx, metric in enumerate(metrics):
                ax = axes[func_idx, metric_idx]
                
                values = []
                for algo_name in algorithm_names:
                    data = results_dict[func_name][algo_name]
                    
                    if metric == 'Mean':
                        values.append(np.mean(data))
                    elif metric == 'Median':
                        values.append(np.median(data))
                    elif metric == 'Std':
                        values.append(np.std(data))
                    elif metric == 'Best':
                        values.append(np.min(data))
                    elif metric == 'Worst':
                        values.append(np.max(data))
                
                # Bar plot
                colors = plt.cm.Set3(np.linspace(0, 1, n_algos))
                bars = ax.bar(algorithm_names, values, color=colors, alpha=0.8)
                
                # Highlight best performer
                if metric != 'Std':
                    best_idx = np.argmin(values)
                    bars[best_idx].set_edgecolor('red')
                    bars[best_idx].set_linewidth(3)
                
                ax.set_title(f'{func_name}\n{metric}', fontsize=10, fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f'{self.save_dir}/{save_name}.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_scalability_analysis(self,
                                 dimensions: List[int],
                                 results: Dict[str, List[float]],
                                 save_name: Optional[str] = None):
        """
        Plot scalability: performance vs problem dimensions.
        
        Args:
            dimensions: List of problem dimensions tested
            results: {algorithm_name: [fitness_values for each dimension]}
            save_name: Filename to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        
        for (algo_name, fitness_values), color in zip(results.items(), colors):
            # Plot 1: Linear scale
            ax1.plot(dimensions, fitness_values, 'o-',
                    label=algo_name, color=color, linewidth=2, markersize=8)
            
            # Plot 2: Log-log scale
            ax2.loglog(dimensions, fitness_values, 'o-',
                      label=algo_name, color=color, linewidth=2, markersize=8)
        
        ax1.set_xlabel('Problem Dimension', fontsize=12)
        ax1.set_ylabel('Final Fitness Value', fontsize=12)
        ax1.set_title('Scalability Analysis\n(Linear Scale)', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Problem Dimension', fontsize=12)
        ax2.set_ylabel('Final Fitness Value', fontsize=12)
        ax2.set_title('Scalability Analysis\n(Log-Log Scale)', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f'{self.save_dir}/{save_name}.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_summary_report(self,
                            all_results: Dict,
                            save_name: str = 'summary_report'):
        """
        Create comprehensive summary report with multiple visualizations.
        
        Args:
            all_results: Dictionary containing all experimental results
            save_name: Base filename for saving
        """
        # Extract data
        convergence_data = all_results.get('convergence', {})
        performance_data = all_results.get('performance', {})
        sensitivity_data = all_results.get('sensitivity', {})
        
        # Plot convergence
        if convergence_data:
            for func_name, algo_results in convergence_data.items():
                self.plot_convergence_curves(
                    algo_results,
                    func_name,
                    save_name=f'{save_name}_convergence_{func_name}'
                )
        
        # Plot performance comparison
        if performance_data:
            self.plot_performance_comparison(
                performance_data,
                save_name=f'{save_name}_performance'
            )
        
        # Plot statistical analysis
        if performance_data:
            self.plot_statistical_analysis(
                performance_data,
                save_name=f'{save_name}_statistics'
            )
        
        print(f"\nSummary report saved to {self.save_dir}/")


# ============= Helper Functions =============

def save_results(results: Dict, filename: str):
    """Save results to pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {filename}")


def load_results(filename: str) -> Dict:
    """Load results from pickle file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)


# ============= Example Usage =============

def example_usage():
    """Demonstrate visualizer usage."""
    visualizer = SwarmVisualizer(save_dir='example_visualizations')
    
    # Example 1: Convergence curves
    print("Example 1: Convergence Curves")
    convergence_data = {
        'PSO': [
            np.exp(-0.1 * np.arange(100)) + np.random.rand(100) * 0.1,
            np.exp(-0.1 * np.arange(100)) + np.random.rand(100) * 0.1,
        ],
        'GA': [
            np.exp(-0.08 * np.arange(100)) + np.random.rand(100) * 0.15,
            np.exp(-0.08 * np.arange(100)) + np.random.rand(100) * 0.15,
        ],
        'ACO': [
            np.exp(-0.09 * np.arange(100)) + np.random.rand(100) * 0.12,
            np.exp(-0.09 * np.arange(100)) + np.random.rand(100) * 0.12,
        ]
    }
    
    visualizer.plot_convergence_curves(
        convergence_data,
        'Sphere Function (10D)',
        save_name='example_convergence'
    )
    
    # Example 2: Performance comparison
    print("\nExample 2: Performance Comparison")
    performance_data = {
        'Sphere-10D': {
            'PSO': np.random.exponential(0.5, 30),
            'GA': np.random.exponential(0.7, 30),
            'ACO': np.random.exponential(0.6, 30),
        },
        'Rastrigin-10D': {
            'PSO': np.random.exponential(5.0, 30),
            'GA': np.random.exponential(7.0, 30),
            'ACO': np.random.exponential(6.0, 30),
        }
    }
    
    visualizer.plot_performance_comparison(
        performance_data,
        save_name='example_performance'
    )
    
    print("\nVisualization examples complete!")


if __name__ == "__main__":
    example_usage()