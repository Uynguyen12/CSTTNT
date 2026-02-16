import numpy as np

def analyze_stability(results_list):
    """
    results_list: list các best_fitness sau n lần chạy (VD: chạy Rastrigin 30 lần)
    """
    mean_fitness = np.mean(results_list)
    std_dev = np.std(results_list)
    best = np.min(results_list)
    worst = np.max(results_list)
    
    return {
        "Mean": mean_fitness,
        "Std Dev (Stability)": std_dev,
        "Best": best,
        "Worst": worst
    }