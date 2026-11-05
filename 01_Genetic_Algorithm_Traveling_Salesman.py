import random
import numpy as np

# Coordinates of cities
cities = {
    'A': (0, 0),
    'B': (1, 5),
    'C': (5, 2),
    'D': (6, 6),
    'E': (8, 3),
    'F': (2, 3)
}

city_names = list(cities.keys())
num_cities = len(city_names)

# --- CORE METRIC FUNCTIONS ---
def calculate_distance(path):
    total_dist = 0
    for i in range(len(path)):
        city1 = cities[path[i]]
        city2 = cities[path[(i + 1) % len(path)]]
        total_dist += np.hypot(city1[0] - city2[0], city1[1] - city2[1])
    return total_dist

def calculate_fitness(path):
    return 1.0 / calculate_distance(path)


# --- OBJECTIVE 1 FUNCTION: ACCELERATE CONVERGENCE SPEED ---
def objective_1_greedy_initialization(pop_size, city_names, cities):
    """
    Objective 1: Speeds up convergence by seeding the initial population 
    with a Nearest-Neighbor heuristic instead of 100% random permutations.
    """
    population = []
    for i in range(pop_size):
        # 40% greedy-seeded individuals, 60% random individuals
        if i < pop_size * 0.4:
            start_city = random.choice(city_names)
            path = [start_city]
            unvisited = set(city_names)
            unvisited.remove(start_city)
            
            while unvisited:
                current = path[-1]
                next_city = min(unvisited, key=lambda c: np.hypot(cities[current][0] - cities[c][0], 
                                                                 cities[current][1] - cities[c][1]))
                path.append(next_city)
                unvisited.remove(next_city)
            population.append(path)
        else:
            ind = city_names[:]
            random.shuffle(ind)
            population.append(ind)
    return population


# --- OBJECTIVE 2 FUNCTION: ENHANCED GREEDY CROSSOVER OPERATOR ---
def objective_2_greedy_crossover(parent1, parent2, cities):
    """
    Objective 2: Enhances search efficiency by letting children choose 
    their next path via the shortest available connecting edges from parents.
    """
    size = len(parent1)
    child = []
    current_city = random.choice([parent1[0], parent2[0]])
    child.append(current_city)
    
    while len(child) < size:
        candidates = []
        for p in [parent1, parent2]:
            try:
                idx = p.index(current_city)
                next_city = p[(idx + 1) % size]
                if next_city not in child:
                    candidates.append(next_city)
            except ValueError:
                continue
                
        if candidates:
            next_city = min(candidates, key=lambda c: np.hypot(cities[current_city][0] - cities[c][0], 
                                                             cities[current_city][1] - cities[c][1]))
        else:
            unvisited = [c for c in cities.keys() if c not in child]
            next_city = min(unvisited, key=lambda c: np.hypot(cities[current_city][0] - cities[c][0], 
                                                             cities[current_city][1] - cities[c][1]))
        child.append(next_city)
        current_city = next_city
        
    return child


# --- OBJECTIVE 3 FUNCTION: LOCAL OPTIMIZATION REFINEMENT ---
def objective_3_local_search(path, cities):
    """
    Objective 3: Implements a 2-opt local search heuristic to fine-tune 
    promising paths locally and prevent the algorithm from getting stuck.
    """
    best_path = list(path)
    best_dist = calculate_distance(best_path)
    improved = True
    
    while improved:
        improved = False
        for i in range(1, len(path) - 1):
            for j in range(i + 1, len(path)):
                new_path = best_path.copy()
                new_path[i:j+1] = reversed(new_path[i:j+1])
                new_dist = calculate_distance(new_path)
                if new_dist < best_dist:
                    best_path = new_path
                    best_dist = new_dist
                    improved = True
        break  # Limit passes to maintain genetic execution speed
    return best_path


# --- SUPPORTING MUTATION OPERATOR ---
def mutate(path, mutation_rate=0.15):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(path)), 2)
        path[i], path[j] = path[j], path[i]
    return path


# --- MAIN GENETIC ALGORITHM ORCHESTRATOR ---
def run_genetic_algorithm(pop_size=50, generations=100):
    # Execute Objective 1: Initialize population with smart seeding
    population = objective_1_greedy_initialization(pop_size, city_names, cities)
    
    for gen in range(generations):
        population = sorted(population, key=lambda x: calculate_fitness(x), reverse=True)
        next_generation = population[:int(pop_size * 0.2)]  # Elitism
        
        while len(next_generation) < pop_size:
            p1, p2 = random.choices(population[:int(pop_size * 0.5)], k=2)
            
            # Execute Objective 2: Generate offspring via Greedy Crossover
            child = objective_2_greedy_crossover(p1, p2, cities)
            child = mutate(child)
            
            # Execute Objective 3: Refine offspring with Local Search optimization
            if random.random() < 0.3:
                child = objective_3_local_search(child, cities)
                
            next_generation.append(child)
            
        population = next_generation
        
    best_path = max(population, key=lambda x: calculate_fitness(x))
    return best_path, calculate_distance(best_path)

# --- EXECUTION ---
best_route, best_distance = run_genetic_algorithm()
print(f"Optimal Path Found: {' -> '.join(best_route)}")
print(f"Total Distance: {best_distance:.2f}")