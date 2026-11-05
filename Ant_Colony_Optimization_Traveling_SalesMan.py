
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ANTS = 25
ITERATIONS = 150
ALPHA = 1.0
BETA = 3.0
EVAPORATION = 0.5
Q = 100.0

def generate_cities(n=25, seed=SEED):
    rng = np.random.default_rng(seed)
    return rng.random((n, 2)) * 100

def calculate_distances(cities):
    difference = cities[:, None, :] - cities[None, :, :]
    distances = np.sqrt(np.sum(difference ** 2, axis=2))
    np.fill_diagonal(distances, 1e-10)
    return distances

def route_length(route, distances):
    return sum(
        distances[route[i], route[(i + 1) % len(route)]]
        for i in range(len(route))
    )

def two_opt(route, distances):
    best = route.copy()
    improved = True

    while improved:
        improved = False

        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                candidate = (
                    best[:i]
                    + best[i:j + 1][::-1]
                    + best[j + 1:]
                )

                if route_length(candidate, distances) < \
                   route_length(best, distances) - 1e-10:
                    best = candidate
                    improved = True

    return best

def ant_colony_optimization(
    distances,
    ants=ANTS,
    iterations=ITERATIONS,
    alpha=ALPHA,
    beta=BETA,
    evaporation=EVAPORATION,
    use_2opt=False,
    seed=SEED
):
    rng = np.random.default_rng(seed)
    n = len(distances)

    pheromone = np.ones((n, n))
    heuristic = 1.0 / distances
    np.fill_diagonal(heuristic, 0)

    best_route = None
    best_distance = float("inf")
    history = []

    for iteration in range(iterations):
        iteration_routes = []

        for _ in range(ants):
            start = int(rng.integers(n))
            route = [start]
            unvisited = set(range(n))
            unvisited.remove(start)

            while unvisited:
                current = route[-1]
                candidates = np.array(sorted(unvisited))

                weights = (
                    pheromone[current, candidates] ** alpha
                    * heuristic[current, candidates] ** beta
                )

                total = weights.sum()

                if not np.isfinite(total) or total <= 0:
                    probabilities = np.ones(len(candidates)) / len(candidates)
                else:
                    probabilities = weights / total

                next_city = int(rng.choice(candidates, p=probabilities))
                route.append(next_city)
                unvisited.remove(next_city)

            if use_2opt:
                route = two_opt(route, distances)

            length = route_length(route, distances)
            iteration_routes.append((route, length))

            if length < best_distance:
                best_route = route.copy()
                best_distance = length

        pheromone *= (1.0 - evaporation)

        for route, length in iteration_routes:
            deposit = Q / max(length, 1e-10)

            for i in range(n):
                a = route[i]
                b = route[(i + 1) % n]
                pheromone[a, b] += deposit
                pheromone[b, a] += deposit

        history.append(best_distance)

    return best_route, best_distance, history

def ordered_crossover(parent1, parent2, rng):
    n = len(parent1)
    left, right = sorted(rng.choice(n, size=2, replace=False))

    child = [-1] * n
    child[left:right + 1] = parent1[left:right + 1]

    remaining = [city for city in parent2 if city not in child]
    positions = [i for i in range(n) if child[i] == -1]

    for position, city in zip(positions, remaining):
        child[position] = city

    return child

def genetic_refinement(
    initial_routes,
    distances,
    generations=40,
    population_size=30,
    mutation_rate=0.15,
    seed=SEED
):
    rng = np.random.default_rng(seed)
    n = len(distances)

    population = [r.copy() for r in initial_routes]

    while len(population) < population_size:
        population.append(rng.permutation(n).tolist())

    population = population[:population_size]

    best_route = min(
        population,
        key=lambda r: route_length(r, distances)
    ).copy()

    best_distance = route_length(best_route, distances)
    history = [best_distance]

    for _ in range(generations):
        population.sort(
            key=lambda r: route_length(r, distances)
        )

        next_population = [r.copy() for r in population[:2]]

        while len(next_population) < population_size:

            def select_parent():
                candidates = rng.choice(
                    len(population), size=3, replace=False
                )
                winner = min(
                    candidates,
                    key=lambda k: route_length(
                        population[int(k)], distances
                    )
                )
                return population[int(winner)]

            p1 = select_parent()
            p2 = select_parent()

            child = ordered_crossover(p1, p2, rng)

            if rng.random() < mutation_rate:
                i, j = rng.choice(n, size=2, replace=False)
                child[i], child[j] = child[j], child[i]

            next_population.append(child)

        population = next_population

        candidate = min(
            population,
            key=lambda r: route_length(r, distances)
        )

        candidate_distance = route_length(candidate, distances)

        if candidate_distance < best_distance:
            best_route = candidate.copy()
            best_distance = candidate_distance

        history.append(best_distance)

    return best_route, best_distance, history

def run_experiments(n_cities=25):
    cities = generate_cities(n_cities)
    distances = calculate_distances(cities)

    results = {}
    routes = {}

    start = time.perf_counter()
    route, distance, history = ant_colony_optimization(
        distances, seed=SEED
    )
    elapsed = time.perf_counter() - start

    results["Standard ACO"] = {
        "Distance": distance,
        "Time (seconds)": elapsed,
        "history": history
    }
    routes["Standard ACO"] = route

    start = time.perf_counter()
    route, distance, history = ant_colony_optimization(
        distances, use_2opt=True, seed=SEED
    )
    elapsed = time.perf_counter() - start

    results["ACO + 2-opt"] = {
        "Distance": distance,
        "Time (seconds)": elapsed,
        "history": history
    }
    routes["ACO + 2-opt"] = route

    start = time.perf_counter()
    rng = np.random.default_rng(SEED)
    initial_routes = []

    for run in range(3):
        candidate, _, _ = ant_colony_optimization(
            distances,
            ants=ANTS,
            iterations=30,
            seed=SEED + 100 + run
        )
        initial_routes.append(candidate)

    for _ in range(15):
        initial_routes.append(rng.permutation(n_cities).tolist())

    route, distance, history = genetic_refinement(
        initial_routes,
        distances,
        generations=40,
        seed=SEED
    )
    elapsed = time.perf_counter() - start

    results["ACO + GA"] = {
        "Distance": distance,
        "Time (seconds)": elapsed,
        "history": history
    }
    routes["ACO + GA"] = route

    summary = pd.DataFrame(results).T[
        ["Distance", "Time (seconds)"]
    ]

    print("\nTSP EXPERIMENT RESULTS")
    print(summary.round(4))
    summary.to_csv("aco_tsp_results.csv")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (name, route) in zip(axes, routes.items()):
        closed_route = route + [route[0]]

        ax.plot(
            cities[closed_route, 0],
            cities[closed_route, 1],
            "-o",
            markersize=4
        )

        ax.set_title(
            f"{name}\nDistance = "
            f"{route_length(route, distances):.2f}"
        )
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("aco_tsp_routes.png", dpi=200)
    plt.show()

    plt.figure(figsize=(9, 5))

    for name in results:
        history = results[name]["history"]
        plt.plot(
            range(1, len(history) + 1),
            history,
            label=name
        )

    plt.xlabel("Iteration / generation")
    plt.ylabel("Best route distance")
    plt.title("Convergence Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("aco_tsp_convergence.png", dpi=200)
    plt.show()

    return cities, distances, results, routes

def parameter_experiment(n_cities=25):
    cities = generate_cities(n_cities)
    distances = calculate_distances(cities)

    records = []

    for evaporation in [0.2, 0.4, 0.6, 0.8]:
        for trial in range(3):
            start = time.perf_counter()

            _, distance, _ = ant_colony_optimization(
                distances,
                ants=ANTS,
                iterations=ITERATIONS,
                evaporation=evaporation,
                seed=SEED + trial
            )

            elapsed = time.perf_counter() - start

            records.append({
                "Evaporation": evaporation,
                "Trial": trial + 1,
                "Best distance": distance,
                "Time (seconds)": elapsed
            })

    df = pd.DataFrame(records)
    df.to_csv("aco_parameter_results.csv", index=False)

    print("\nPARAMETER SENSITIVITY RESULTS")
    print(
        df.groupby("Evaporation")[
            ["Best distance", "Time (seconds)"]
        ].agg(["mean", "std"]).round(4)
    )

    return df

if __name__ == "__main__":
    cities, distances, results, routes = run_experiments(
        n_cities=25
    )

    parameter_experiment(n_cities=25)

    print("\nFiles generated:")
    print("1. aco_tsp_results.csv")
    print("2. aco_tsp_routes.png")
    print("3. aco_tsp_convergence.png")
    print("4. aco_parameter_results.csv")
