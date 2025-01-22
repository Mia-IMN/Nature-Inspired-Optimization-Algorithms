from shutil import which

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import ticker
from numpy import ma


# Particle Swarm Optimization (PSO)
class PSO:
    def __init__(self, n_particles=30, dimensions=2, iterations=100):
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.iterations = iterations
        self.best_global_position = None
        self.best_global_value = float('inf')
        self.particles = np.random.uniform(-5, 5, (n_particles, dimensions))
        self.velocities = np.random.uniform(-1, 1, (n_particles, dimensions))
        self.best_particle_positions = self.particles.copy()
        self.best_particle_values = np.array([self.objective_function(p) for p in self.particles])

    @staticmethod
    def objective_function(x):
        return sum([xi**2 - 10 * np.cos(2 * np.pi * xi) + 10 for xi in x])

    def optimize(self):
        for _ in range(self.iterations):
            for i, particle in enumerate(self.particles):
                fitness = self.objective_function(particle)
                if fitness < self.best_particle_values[i]:
                    self.best_particle_values[i] = fitness
                    self.best_particle_positions[i] = particle

                if fitness < self.best_global_value:
                    self.best_global_value = fitness
                    self.best_global_position = particle

            self.velocities = (
                0.5 * self.velocities +
                2 * np.random.rand(self.n_particles, self.dimensions) * (self.best_particle_positions - self.particles) +
                2 * np.random.rand(self.n_particles, self.dimensions) * (self.best_global_position - self.particles)
            )
            self.particles += self.velocities

        return self.best_global_value

    def visualize(self):
        fig, ax = plt.subplots()
        X = np.linspace(-5, 5, 100)
        Y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(X, Y)
        Z = self.objective_function([X, Y])

        # Z1 = np.exp(-(X) ** 2 - (Y) ** 2)
        # z = 50 * Z1
        # z[:5, :5] = -1
        # z = ma.masked_where(z <= 0, z)
        # ax.contourf(X, Y, z, locator=ticker.LogLocator(), cmap="Greens")
        scatter = ax.scatter(self.particles[:, 0], self.particles[:, 1], c='red', marker='o')

        def update(frame):
            self.optimize_step()
            scatter.set_offsets(self.particles)
            return scatter,

        ani = animation.FuncAnimation(fig, update, frames=self.iterations, interval=100, blit=True)
        plt.grid(which='major', color='k', linestyle='-', alpha=0.5)
        plt.grid(which='minor', color='r', linestyle='-', alpha=0.2)
        plt.minorticks_on()
        plt.title("PSO Animated Visual")
        plt.show()

    def optimize_step(self):
        for i, particle in enumerate(self.particles):
            fitness = self.objective_function(particle)
            if fitness < self.best_particle_values[i]:
                self.best_particle_values[i] = fitness
                self.best_particle_positions[i] = particle

            if fitness < self.best_global_value:
                self.best_global_value = fitness
                self.best_global_position = particle

        self.velocities = (
            0.5 * self.velocities +
            2 * np.random.rand(self.n_particles, self.dimensions) * (self.best_particle_positions - self.particles) +
            2 * np.random.rand(self.n_particles, self.dimensions) * (self.best_global_position - self.particles)
        )
        self.particles += self.velocities


if __name__ == "__main__":
    # Instantiate and run the PSO algorithm with visualization
    pso = PSO(n_particles=30, dimensions=2, iterations=50)
    pso.visualize()
