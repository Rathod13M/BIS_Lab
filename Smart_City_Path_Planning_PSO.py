

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean
import time

np.random.seed(42)

def sphere(x):
    return np.sum(x**2, axis=-1)

def rastrigin(x):
    return 10 * x.shape[-1] + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=-1)

def ackley(x):
    d = x.shape[-1]
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2, axis=-1) / d)) - np.exp(np.sum(np.cos(2 * np.pi * x), axis=-1) / d) + 20 + np.e

def rosenbrock(x):
    return np.sum(100 * (x[..., 1:] - x[..., :-1]**2)**2 + (1 - x[..., :-1])**2, axis=-1)

def griewank(x):
    i = np.arange(1, x.shape[-1] + 1)
    return 1 + np.sum(x**2, axis=-1) / 4000 - np.prod(np.cos(x / np.sqrt(i)), axis=-1)

def schwefel(x):
    return 418.9829 * x.shape[-1] - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=-1)

def objective(x, name):
    functions = {
        "Sphere": sphere,
        "Rastrigin": rastrigin,
        "Ackley": ackley,
        "Rosenbrock": rosenbrock,
        "Griewank": griewank,
        "Schwefel": schwefel
    }
    return functions[name](x)

def initialize(n, dim, bounds):
    low, high = bounds
    return np.random.uniform(low, high, (n, dim))

def pso(func, dim, bounds, n=30, iterations=300):
    x = initialize(n, dim, bounds)
    v = np.zeros_like(x)
    pbest = x.copy()
    pfit = func(x)
    gbest = pbest[np.argmin(pfit)].copy()
    gfit = np.min(pfit)
    history = []
    for t in range(iterations):
        w = 0.9 - 0.5 * t / iterations
        r1, r2 = np.random.rand(n, dim), np.random.rand(n, dim)
        v = w * v + 2 * r1 * (pbest - x) + 2 * r2 * (gbest - x)
        x = np.clip(x + v, bounds[0], bounds[1])
        fit = func(x)
        mask = fit < pfit
        pbest[mask] = x[mask]
        pfit[mask] = fit[mask]
        if np.min(pfit) < gfit:
            gfit = np.min(pfit)
            gbest = pbest[np.argmin(pfit)].copy()
        history.append(gfit)
    return gbest, gfit, history

def qpso(func, dim, bounds, n=30, iterations=300):
    x = initialize(n, dim, bounds)
    pbest = x.copy()
    pfit = func(x)
    gbest = pbest[np.argmin(pfit)].copy()
    gfit = np.min(pfit)
    history = []
    for t in range(iterations):
        mbest = np.mean(pbest, axis=0)
        beta = 1.0 - 0.5 * t / iterations
        phi = np.random.rand(n, dim)
        attractor = phi * pbest + (1 - phi) * gbest
        u = np.maximum(np.random.rand(n, dim), 1e-12)
        direction = np.where(np.random.rand(n, dim) < 0.5, 1, -1)
        x = attractor + direction * beta * np.abs(mbest - x) * np.log(1 / u)
        x = np.clip(x, bounds[0], bounds[1])
        fit = func(x)
        mask = fit < pfit
        pbest[mask] = x[mask]
        pfit[mask] = fit[mask]
        if np.min(pfit) < gfit:
            gfit = np.min(pfit)
            gbest = pbest[np.argmin(pfit)].copy()
        history.append(gfit)
    return gbest, gfit, history

def ltqpso(func, dim, bounds, n=30, iterations=300):
    x = initialize(n, dim, bounds)
    pbest = x.copy()
    pfit = func(x)
    gbest = pbest[np.argmin(pfit)].copy()
    gfit = np.min(pfit)
    history = []
    for t in range(iterations):
        mbest = np.mean(pbest, axis=0)
        beta = 1.0 - 0.5 * t / iterations
        spread = np.std(pbest, axis=0) / (np.std(x, axis=0) + 1e-12)
        spread = np.clip(spread, 0.2, 2.0)
        progress = np.clip(pfit / (np.max(pfit) + 1e-12), 0, 1)
        phi = np.random.rand(n, dim)
        attractor = phi * pbest + (1 - phi) * gbest
        u = np.maximum(np.random.rand(n, dim), 1e-12)
        direction = np.where(np.random.rand(n, dim) < 0.5, 1, -1)
        scale = beta * (0.5 + progress[:, None]) * spread
        candidate = attractor + direction * scale * np.abs(mbest - x) * np.log(1 / u)
        candidate = np.clip(candidate, bounds[0], bounds[1])
        candidate_fit = func(candidate)
        combined = np.concatenate((x, candidate), axis=0)
        combined_fit = np.concatenate((func(x), candidate_fit))
        selected = np.argsort(combined_fit)[:n]
        x = combined[selected].copy()
        fit = combined_fit[selected]
        mask = fit < pfit
        pbest[mask] = x[mask]
        pfit[mask] = fit[mask]
        if np.min(pfit) < gfit:
            gfit = np.min(pfit)
            gbest = pbest[np.argmin(pfit)].copy()
        history.append(gfit)
    return gbest, gfit, history

def compare_algorithms():
    names = ["Sphere", "Rastrigin", "Ackley", "Rosenbrock", "Griewank", "Schwefel"]
    algorithms = {"PSO": pso, "QPSO": qpso, "LTQPSO": ltqpso}
    results = {}
    for name in names:
        bounds = (-5.12, 5.12)
        if name == "Rosenbrock":
            bounds = (-2.048, 2.048)
        elif name == "Griewank":
            bounds = (-600, 600)
        elif name == "Schwefel":
            bounds = (-500, 500)
        func = lambda x, nm=name: objective(x, nm)
        results[name] = {}
        plt.figure(figsize=(8, 5))
        for algorithm_name, algorithm in algorithms.items():
            start = time.perf_counter()
            best_x, best_f, history = algorithm(func, 10, bounds)
            elapsed = time.perf_counter() - start
            results[name][algorithm_name] = (best_f, elapsed, best_x)
            plt.plot(history, label=algorithm_name)
            print(name, algorithm_name, "Best fitness:", best_f, "Time:", round(elapsed, 4), "Best position:", best_x)
        plt.title(name + " Convergence")
        plt.xlabel("Iteration")
        plt.ylabel("Best objective value")
        plt.yscale("symlog", linthresh=1e-10)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    return results

def function_approximation():
    x = np.linspace(-3, 3, 300).reshape(-1, 1)
    y = np.sin(x[:, 0]) + 0.2 * x[:, 0]**2
    def error_function(params):
        a, b, c, d = params
        predicted = a * np.sin(b * x[:, 0]) + c * x[:, 0]**2 + d
        return np.mean((predicted - y)**2)
    best, error, history = ltqpso(error_function, 4, (-5, 5), n=40, iterations=500)
    predicted = best[0] * np.sin(best[1] * x[:, 0]) + best[2] * x[:, 0]**2 + best[3]
    print("Function approximation parameters:", best)
    print("Function approximation MSE:", error)
    plt.figure(figsize=(8, 5))
    plt.plot(x[:, 0], y, label="Target function")
    plt.plot(x[:, 0], predicted, "--", label="LTQPSO approximation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(8, 5))
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Mean squared error")
    plt.title("Function Approximation Convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def landscape_path_planning():
    obstacles = [
        (20, 20, 7),
        (35, 45, 8),
        (55, 30, 10),
        (70, 65, 9),
        (45, 75, 7)
    ]
    start = np.array([5.0, 5.0])
    goal = np.array([95.0, 95.0])
    n_waypoints = 8
    dim = n_waypoints * 2

    def decode(params):
        return params.reshape(n_waypoints, 2)

    def path_objective(params):
        waypoints = decode(params)
        path = np.vstack([start, waypoints, goal])
        lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        cost = np.sum(lengths)
        for px, py, radius in obstacles:
            distances = np.linalg.norm(path - np.array([px, py]), axis=1)
            penetration = np.maximum(radius + 3 - distances, 0)
            cost += 1000 * np.sum(penetration**2)
            for i in range(len(path) - 1):
                segment = path[i + 1] - path[i]
                denom = np.dot(segment, segment) + 1e-12
                t = np.clip(np.dot(np.array([px, py]) - path[i], segment) / denom, 0, 1)
                nearest = path[i] + t * segment
                distance = np.linalg.norm(nearest - np.array([px, py]))
                cost += 1000 * max(radius + 3 - distance, 0)**2
        return cost

    best, cost, history = ltqpso(path_objective, dim, (0, 100), n=60, iterations=600)
    path = np.vstack([start, decode(best), goal])
    print("Landscape trail path cost:", cost)
    print("Landscape trail path length:", np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    plt.figure(figsize=(8, 8))
    for px, py, radius in obstacles:
        circle = plt.Circle((px, py), radius, color="red", alpha=0.3)
        plt.gca().add_patch(circle)
        safety_circle = plt.Circle((px, py), radius + 3, color="red", fill=False, linestyle="--")
        plt.gca().add_patch(safety_circle)
    plt.plot(path[:, 0], path[:, 1], "-o", color="blue", label="Optimized trail")
    plt.scatter(*start, color="green", s=100, label="Start")
    plt.scatter(*goal, color="purple", s=100, label="Goal")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title("Landscape Trail Path Planning")
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(8, 5))
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Best path cost")
    plt.title("Landscape Path Planning Convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_algorithms()
    function_approximation()
    landscape_path_planning()