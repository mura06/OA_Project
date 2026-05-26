import numpy as np
from optimization_lib.network import fitness_function

def init_uniform(model, rng):
    """
    Generates a random weight vector using Xavier/Glorot uniform initialization for each layer.
    """
    parts = []
    for i in range(len(model.coefs_)):
        n_in, n_out = model.coefs_[i].shape
        limit = np.sqrt(6.0 / (n_in + n_out))
        parts.append(rng.uniform(-limit, limit, size=n_in * n_out))
        parts.append(rng.uniform(-limit, limit, size=n_out))
    return np.concatenate(parts)

def init_normal(model, rng):
    """
    Generates a random weight vector using He normal initialization for each layer.
    """
    parts = []
    for i in range(len(model.coefs_)):
        n_in, n_out = model.coefs_[i].shape
        std = np.sqrt(2.0 / n_in)
        parts.append(rng.normal(0, std, size=n_in * n_out))
        parts.append(rng.normal(0, std, size=n_out))
    return np.concatenate(parts)

def tournament_selection(population, fitnesses, tournament_size, rng):
    """
    Selects the best of K randomly selected individuals.
    """
    selected_idx = rng.choice(len(population), size=tournament_size, replace=False)
    best_idx = selected_idx[np.argmax(fitnesses[selected_idx])]
    return population[best_idx].copy()

def roulette_wheel_selection(population, fitnesses, rng):
    """
    Rank-based roulette wheel selection to handle negative fitness values and maintain stability.
    """
    ranks = np.argsort(np.argsort(fitnesses))  # ranks from 0 (worst) to N-1 (best)
    probs = (ranks + 1) / np.sum(ranks + 1)
    idx = rng.choice(len(population), p=probs)
    return population[idx].copy()

def arithmetic_crossover(parent1, parent2, rng):
    """
    Blends parents linearly: c = beta * p1 + (1 - beta) * p2.
    """
    beta = rng.uniform(0, 1)
    child1 = beta * parent1 + (1 - beta) * parent2
    child2 = (1 - beta) * parent1 + beta * parent2
    return child1, child2

def blx_alpha_crossover(parent1, parent2, alpha, rng):
    """
    Blend Crossover (BLX-alpha) creates offspring in an expanded range between parents.
    """
    d = np.abs(parent1 - parent2)
    low = np.minimum(parent1, parent2) - alpha * d
    high = np.maximum(parent1, parent2) + alpha * d
    child1 = rng.uniform(low, high)
    child2 = rng.uniform(low, high)
    return child1, child2

def gaussian_mutation(individual, mutation_rate, scale, rng):
    """
    Adds Gaussian noise N(0, scale^2) to mutated genes.
    """
    mutated = individual.copy()
    mask = rng.uniform(0, 1, size=len(individual)) < mutation_rate
    mutated[mask] += rng.normal(0, scale, size=np.sum(mask))
    return mutated

def uniform_mutation(individual, mutation_rate, scale, rng):
    """
    Adds uniform noise in range [-scale, scale] to mutated genes.
    """
    mutated = individual.copy()
    mask = rng.uniform(0, 1, size=len(individual)) < mutation_rate
    mutated[mask] += rng.uniform(-scale, scale, size=np.sum(mask))
    return mutated


class GeneticAlgorithm:
    def __init__(self, model, X, y, pop_size=50, generations=100, 
                 crossover_prob=0.8, mutation_prob=0.2, mutation_gene_rate=0.1,
                 elitism_count=2, init_method="uniform", selection_method="tournament",
                 crossover_method="blx_alpha", mutation_method="gaussian",
                 tournament_size=3, blx_alpha=0.5, mutation_scale=0.1,
                 metric="f1_macro", random_state=42):
        self.model = model
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.mutation_gene_rate = mutation_gene_rate
        self.elitism_count = elitism_count
        self.init_method = init_method
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.tournament_size = tournament_size
        self.blx_alpha = blx_alpha
        self.mutation_scale = mutation_scale
        self.metric = metric
        self.rng = np.random.default_rng(random_state)
        
        # Determine number of network parameters
        self.num_params = len(model.coefs_[0].flatten()) + len(model.intercepts_[0])
        # Find exact size
        dummy_weights = []
        for coef, intercept in zip(model.coefs_, model.intercepts_):
            dummy_weights.append(coef.flatten())
            dummy_weights.append(intercept.flatten())
        self.num_params = len(np.concatenate(dummy_weights))

    def solve(self):
        # 1. Initialize population
        population = []
        for _ in range(self.pop_size):
            if self.init_method == "uniform":
                ind = init_uniform(self.model, self.rng)
            else:
                ind = init_normal(self.model, self.rng)
            population.append(ind)
        population = np.array(population)

        # Track history
        best_fitness_history = []
        mean_fitness_history = []
        best_individual = None
        best_fitness = -np.inf

        # 2. Optimization loop
        for gen in range(self.generations):
            # Evaluate fitness of all individuals
            fitnesses = np.array([
                fitness_function(ind, self.model, self.X, self.y, self.metric)
                for ind in population
            ])

            # Track best
            current_best_idx = np.argmax(fitnesses)
            current_best_fitness = fitnesses[current_best_idx]
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[current_best_idx].copy()
            
            best_fitness_history.append(best_fitness)
            mean_fitness_history.append(np.mean(fitnesses))

            # Create next generation
            next_generation = []

            # Elitism
            sorted_indices = np.argsort(fitnesses)[::-1]
            for i in range(min(self.elitism_count, self.pop_size)):
                next_generation.append(population[sorted_indices[i]].copy())

            # Breed remaining individuals
            while len(next_generation) < self.pop_size:
                # Selection
                if self.selection_method == "tournament":
                    parent1 = tournament_selection(population, fitnesses, self.tournament_size, self.rng)
                    parent2 = tournament_selection(population, fitnesses, self.tournament_size, self.rng)
                else:
                    parent1 = roulette_wheel_selection(population, fitnesses, self.rng)
                    parent2 = roulette_wheel_selection(population, fitnesses, self.rng)

                # Crossover
                if self.rng.uniform(0, 1) < self.crossover_prob:
                    if self.crossover_method == "blx_alpha":
                        child1, child2 = blx_alpha_crossover(parent1, parent2, self.blx_alpha, self.rng)
                    else:
                        child1, child2 = arithmetic_crossover(parent1, parent2, self.rng)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Mutation
                if self.rng.uniform(0, 1) < self.mutation_prob:
                    if self.mutation_method == "gaussian":
                        child1 = gaussian_mutation(child1, self.mutation_gene_rate, self.mutation_scale, self.rng)
                    else:
                        child1 = uniform_mutation(child1, self.mutation_gene_rate, self.mutation_scale, self.rng)
                
                if self.rng.uniform(0, 1) < self.mutation_prob:
                    if self.mutation_method == "gaussian":
                        child2 = gaussian_mutation(child2, self.mutation_gene_rate, self.mutation_scale, self.rng)
                    else:
                        child2 = uniform_mutation(child2, self.mutation_gene_rate, self.mutation_scale, self.rng)

                next_generation.append(child1)
                if len(next_generation) < self.pop_size:
                    next_generation.append(child2)

            population = np.array(next_generation)

        # Final evaluation
        fitnesses = np.array([
            fitness_function(ind, self.model, self.X, self.y, self.metric)
            for ind in population
        ])
        current_best_idx = np.argmax(fitnesses)
        current_best_fitness = fitnesses[current_best_idx]
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[current_best_idx].copy()

        return best_individual, best_fitness, best_fitness_history, mean_fitness_history
