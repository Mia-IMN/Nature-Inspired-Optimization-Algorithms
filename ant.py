import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Ant Colony Optimization (ACO)
class AntColony:
    def __init__(self, n_ants=30, n_iterations=100, alpha=1, beta=2, evaporation_rate=0.5, n_nodes=10):
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.n_nodes = n_nodes
        self.distances = np.random.randint(1, 20, size=(n_nodes, n_nodes))
        np.fill_diagonal(self.distances, 0)
        self.pheromones = np.ones((n_nodes, n_nodes))

    def objective_function(self, path):
        return sum([self.distances[path[i], path[i + 1]] for i in range(len(path) - 1)])

    def optimize(self):
        best_path = None
        best_path_length = float('inf')

        for _ in range(self.n_iterations):
            all_paths = []
            for _ in range(self.n_ants):
                path = self.construct_path()
                path_length = self.objective_function(path)
                all_paths.append((path, path_length))

                if path_length < best_path_length:
                    best_path_length = path_length
                    best_path = path

            self.update_pheromones(all_paths)

        return best_path, best_path_length

    def construct_path(self):
        path = [np.random.randint(self.n_nodes)]
        while len(path) < self.n_nodes:
            current_node = path[-1]
            probabilities = self.calculate_transition_probabilities(current_node, path)
            next_node = np.random.choice(range(self.n_nodes), p=probabilities)
            path.append(next_node)
        path.append(path[0])  # Return to the starting node
        return path

    def calculate_transition_probabilities(self, current_node, visited):
        probabilities = []
        for next_node in range(self.n_nodes):
            if next_node not in visited:
                pheromone = self.pheromones[current_node, next_node] ** self.alpha
                heuristic = (1 / self.distances[current_node, next_node]) ** self.beta
                probabilities.append(pheromone * heuristic)
            else:
                probabilities.append(0)
        probabilities = np.array(probabilities)
        return probabilities / probabilities.sum()

    def update_pheromones(self, all_paths):
        self.pheromones *= (1 - self.evaporation_rate)
        for path, path_length in all_paths:
            for i in range(len(path) - 1):
                self.pheromones[path[i], path[i + 1]] += 1 / path_length

    def visualize(self):
        fig, ax = plt.subplots()
        positions = np.random.rand(self.n_nodes, 2) * 10
        ax.scatter(positions[:, 0], positions[:, 1], c='blue', marker='o')

        lines = []
        for _ in range(self.n_ants):
            line, = ax.plot([], [], c='red', alpha=0.5)
            lines.append(line)

        def update(frame):
            for i, line in enumerate(lines):
                path = self.construct_path()
                line.set_data(positions[path, 0], positions[path, 1])
            return lines

        ani = animation.FuncAnimation(fig, update, frames=self.n_iterations, interval=100, blit=True)
        plt.show()

if __name__ == "__main__":
    aco = AntColony()
    aco.visualize()
