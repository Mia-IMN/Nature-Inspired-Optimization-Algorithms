import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Genetic Algorithm (GA)
class GeneticAlgorithm:
    def __init__(self, population_size=30, dimensions=2, iterations=100):
        self.population_size = population_size
        self.dimensions = dimensions
        self.iterations = iterations
        self.mutation_rate = 0.1
        self.population = np.random.uniform(-5, 5, (population_size, dimensions))
        self.fitness = np.array([self.objective_function(ind) for ind in self.population])

    @staticmethod
    def objective_function(x):
        return sum([xi**2 - 10 * np.cos(2 * np.pi * xi) + 10 for xi in x])

    def optimize(self):
        for _ in range(self.iterations):
            new_population = []
            for _ in range(self.population_size // 2):
                parents = self.select_parents()
                offspring = self.crossover(parents)
                new_population.extend([self.mutate(child) for child in offspring])
            self.population = np.array(new_population)
            self.fitness = np.array([self.objective_function(ind) for ind in self.population])
        return np.min(self.fitness)

    def visualize(self):
        fig, ax = plt.subplots()
        X = np.linspace(-5, 5, 100)
        Y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(X, Y)
        Z = self.objective_function([X, Y])

        scatter = ax.scatter(self.population[:, 0], self.population[:, 1], c='red', marker='o')

        def update(frame):
            self.optimize_step()
            scatter.set_offsets(self.population)
            return scatter,

        ani = animation.FuncAnimation(fig, update, frames=self.iterations, interval=100, blit=True)
        plt.grid(which='major', color='k', linestyle='-', alpha=0.5)
        plt.grid(which='minor', color='r', linestyle='-', alpha=0.2)
        plt.minorticks_on()
        plt.title("Genetic Algorithm Animated Visual")
        plt.show()

    def optimize_step(self):
        new_population = []
        for _ in range(self.population_size // 2):
            parents = self.select_parents()
            offspring = self.crossover(parents)
            new_population.extend([self.mutate(child) for child in offspring])
        self.population = np.array(new_population)
        self.fitness = np.array([self.objective_function(ind) for ind in self.population])

    def select_parents(self):
        fitness_probs = 1 / (1 + self.fitness)
        fitness_probs /= fitness_probs.sum()
        parents = self.population[np.random.choice(self.population_size, size=2, p=fitness_probs)]
        return parents

    def crossover(self, parents):
        crossover_point = np.random.randint(1, self.dimensions)
        child1 = np.concatenate((parents[0][:crossover_point], parents[1][crossover_point:]))
        child2 = np.concatenate((parents[1][:crossover_point], parents[0][crossover_point:]))
        return child1, child2

    def mutate(self, individual):
        if np.random.rand() < self.mutation_rate:
            mutation_index = np.random.randint(0, self.dimensions)
            individual[mutation_index] += np.random.uniform(-1, 1)
        return individual


if __name__ == "__main__":
    goa = GeneticAlgorithm()
    goa.visualize()