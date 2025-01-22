import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Firefly Algorithm (FA)
class Firefly:
    def __init__(self, n_fireflies=30, dimensions=2, iterations=100):
        self.n_fireflies = n_fireflies
        self.dimensions = dimensions
        self.iterations = iterations
        self.gamma = 1.0  # Light absorption coefficient
        self.beta0 = 2.0  # Maximum attractiveness
        self.alpha = 0.2  # Randomness scaling
        self.fireflies = np.random.uniform(-5, 5, (n_fireflies, dimensions))
        self.light_intensity = np.array([self.objective_function(f) for f in self.fireflies])

    @staticmethod
    def objective_function(x):
        return sum([xi**2 - 10 * np.cos(2 * np.pi * xi) + 10 for xi in x])

    def optimize(self):
        for _ in range(self.iterations):
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    if self.light_intensity[j] < self.light_intensity[i]:
                        distance = np.linalg.norm(self.fireflies[i] - self.fireflies[j])
                        beta = self.beta0 * np.exp(-self.gamma * distance**2)
                        self.fireflies[i] += beta * (self.fireflies[j] - self.fireflies[i]) + self.alpha * np.random.uniform(-1, 1, self.dimensions)
                        self.light_intensity[i] = self.objective_function(self.fireflies[i])
        return np.min(self.light_intensity)

    def visualize(self):
        fig, ax = plt.subplots()
        X = np.linspace(-5, 5, 100)
        Y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(X, Y)
        Z = self.objective_function([X, Y])

        scatter = ax.scatter(self.fireflies[:, 0], self.fireflies[:, 1], c='red', marker='o')

        def update(frame):
            self.optimize_step()
            scatter.set_offsets(self.fireflies)
            return scatter,

        ani = animation.FuncAnimation(fig, update, frames=self.iterations, interval=100, blit=True)
        plt.grid(which='major', color='k', linestyle='-', alpha=0.5)
        plt.grid(which='minor', color='r', linestyle='-', alpha=0.2)
        plt.minorticks_on()
        plt.title("Firefly Algorithm Animated Visual")
        plt.show()

    def optimize_step(self):
        for i in range(self.n_fireflies):
            for j in range(self.n_fireflies):
                if self.light_intensity[j] < self.light_intensity[i]:
                    distance = np.linalg.norm(self.fireflies[i] - self.fireflies[j])
                    beta = self.beta0 * np.exp(-self.gamma * distance**2)
                    self.fireflies[i] += beta * (self.fireflies[j] - self.fireflies[i]) + self.alpha * np.random.uniform(-1, 1, self.dimensions)
                    self.light_intensity[i] = self.objective_function(self.fireflies[i])

if __name__ == "__main__":
    foa = Firefly()
    foa.visualize()