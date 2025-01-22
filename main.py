import time
import matplotlib.pyplot as plt
import numpy as np

# Importing algorithms from the modules
from pso import PSO
from ant import AntColony
from firefly import Firefly
from genetic import GeneticAlgorithm

# Define the objective function
def objective_function(x):
    return sum([xi**2 - 10 * np.cos(2 * np.pi * xi) + 10 for xi in x])

def run_algorithm(algorithm, name):
    start_time = time.time()
    best_result = algorithm.optimize()
    execution_time = time.time() - start_time
    return name, execution_time, best_result

def main():
    # Parameters for the optimization
    dimensions = 2
    iterations = 100
    n_particles = 30

    # Initialize algorithms
    algorithms = [
        (PSO(n_particles, dimensions, iterations), "PSO"),
        (AntColony(n_ants=n_particles), "Ant Colony"),
        (Firefly(n_fireflies=n_particles, dimensions=dimensions, iterations=iterations), "Firefly"),
        (GeneticAlgorithm(population_size=n_particles, dimensions=dimensions, iterations=iterations), "Genetic Algorithm"),
    ]

    results = []

    # Run and compare algorithms
    for algorithm, name in algorithms:
        name, execution_time, best_result = run_algorithm(algorithm, name)
        if name == "Ant Colony":
            best_result = best_result[1]  # Extract the numeric cost from the tuple
        results.append((name, execution_time, best_result))
        print(f"{name} - Best Result: {best_result}, Execution Time: {execution_time:.4f}s")

    # Visualization
    algorithm_names = [result[0] for result in results]
    execution_times = [result[1] for result in results]
    best_results = [result[2] for result in results]

    # Bar chart for execution time
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.bar(algorithm_names, execution_times, color=['blue', 'green', 'red', 'purple'])
    plt.title('Execution Time Comparison')
    plt.ylabel('Time (s)')
    plt.xlabel('Algorithm')

    # Line graph for optimization results
    plt.subplot(1, 2, 2)
    plt.plot(algorithm_names, best_results, marker='o', linestyle='-', color='orange')
    plt.title('Best Results Comparison')
    plt.ylabel('Best Objective Value')
    plt.xlabel('Algorithm')
    plt.grid(which='major', color='k', linestyle='-', alpha=0.5)
    plt.grid(which='minor', color='r', linestyle='-', alpha=0.2)
    plt.minorticks_on()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
