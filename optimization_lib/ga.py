import numpy as np
from optimization_lib.network import fitness_function, generate_solution


def tournament_selection(population, fitnesses, tournament_size, rng):
    """
    Selects the best of K randomly selected individuals.
    
    Parameters:
        population: The array of candidate solutions.
        fitnesses: The fitness scores of the population.
        tournament_size: The number of individuals to sample.
        rng: The numpy random generator.
    
    Returns:
        np.ndarray: The selected individual (copied).
    """
    # Randomly select K individuals for the tournament
    selected_idx = rng.choice(len(population), size=tournament_size, replace=False)
    # Find the index of the individual with the highest fitness among the selected
    best_idx = selected_idx[np.argmax(fitnesses[selected_idx])]
    # Return a copy of the winning individual
    return population[best_idx].copy()

def roulette_wheel_selection(population, fitnesses, rng):
    """
    Rank-based roulette wheel selection for maintaining stability.
    
    Parameters:
        population: The array of candidate solutions.
        fitnesses: The fitness scores of the population.
        rng: The numpy random generator.
    
    Returns:
        np.ndarray: The selected individual (copied).
    """
    # Calculate ranks from 0 (worst) to N-1 (best)
    ranks = np.argsort(np.argsort(fitnesses))
    # Calculate selection probabilities proportional to rank
    probs = (ranks + 1) / np.sum(ranks + 1)
    # Select one individual based on the calculated probabilities
    idx = rng.choice(len(population), p=probs)
    # Return a copy of the selected individual
    return population[idx].copy()

def arithmetic_crossover(parent1, parent2, rng):
    """
    Blends parents linearly: c = beta * p1 + (1 - beta) * p2.
    
    Parameters:
        parent1: The first parent solution vector.
        parent2: The second parent solution vector.
        rng: The numpy random generator.
    
    Returns:
        tuple: Two new child solution vectors.
    """
    # Generate a random blending factor beta between 0 and 1
    beta = rng.uniform(0, 1)
    # Create the first child by blending parent1 and parent2
    child1 = beta * parent1 + (1 - beta) * parent2
    # Create the second child with the inverse blending ratio
    child2 = (1 - beta) * parent1 + beta * parent2
    # Return the two new children
    return child1, child2

def blx_alpha_crossover(parent1, parent2, alpha, rng):
    """
    Blend Crossover (BLX-alpha) creates offspring in an expanded range between parents.
    
    Parameters:
        parent1: The first parent solution vector.
        parent2: The second parent solution vector.
        alpha: The expansion factor.
        rng: The numpy random generator.
    
    Returns:
        tuple: Two new child solution vectors.
    """
    # Calculate the absolute difference between the two parents
    d = np.abs(parent1 - parent2)
    # Define the lower bound for the extended search space
    low = np.minimum(parent1, parent2) - alpha * d
    # Define the upper bound for the extended search space
    high = np.maximum(parent1, parent2) + alpha * d
    # Generate the first child randomly within the new bounds
    child1 = rng.uniform(low, high)
    # Generate the second child randomly within the new bounds
    child2 = rng.uniform(low, high)
    # Return the two new children
    return child1, child2

def gaussian_mutation(individual, mutation_rate, scale, rng):
    """
    Adds Gaussian noise N(0, scale^2) to mutated genes.
    
    Parameters:
        individual: The solution vector to mutate.
        mutation_rate: The probability of mutating each gene.
        scale: The standard deviation of the Gaussian noise.
        rng: The numpy random generator.
    
    Returns:
        np.ndarray: The mutated solution vector.
    """
    # Create a copy of the individual to mutate
    mutated = individual.copy()
    # Generate a boolean mask to determine which genes will mutate
    mask = rng.uniform(0, 1, size=len(individual)) < mutation_rate
    # Add Gaussian noise to the selected genes
    mutated[mask] += rng.normal(0, scale, size=np.sum(mask))
    # Return the mutated individual
    return mutated

def uniform_mutation(individual, mutation_rate, scale, rng):
    """
    Adds uniform noise in range [-scale, scale] to mutated genes.
    
    Parameters:
        individual: The solution vector to mutate.
        mutation_rate: The probability of mutating each gene.
        scale: The range boundary for uniform noise.
        rng: The numpy random generator.
    
    Returns:
        np.ndarray: The mutated solution vector.
    """
    # Create a copy of the individual to mutate
    mutated = individual.copy()
    # Generate a boolean mask to determine which genes will mutate
    mask = rng.uniform(0, 1, size=len(individual)) < mutation_rate
    # Add uniform noise to the selected genes
    mutated[mask] += rng.uniform(-scale, scale, size=np.sum(mask))
    # Return the mutated individual
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
        
        # Determine exact number of network parameters
        dummy_weights = []
        for coef, intercept in zip(model.coefs_, model.intercepts_):
            dummy_weights.append(coef.flatten())
            dummy_weights.append(intercept.flatten())
        self.num_params = len(np.concatenate(dummy_weights))

    def solve(self):
        # Initialize population
        population = []
        for _ in range(self.pop_size):
            ind = generate_solution(self.model, self.init_method, self.rng)
            population.append(ind)
        population = np.array(population)

        # Track history
        best_fitness_history = []
        mean_fitness_history = []
        best_individual = None
        best_fitness = -np.inf

        # Optimization loop
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

                # Mutation for child 1
                if self.rng.uniform(0, 1) < self.mutation_prob:
                    if self.mutation_method == "gaussian":
                        child1 = gaussian_mutation(child1, self.mutation_gene_rate, self.mutation_scale, self.rng)
                    else:
                        child1 = uniform_mutation(child1, self.mutation_gene_rate, self.mutation_scale, self.rng)
                
                # Mutation for child 2
                if self.rng.uniform(0, 1) < self.mutation_prob:
                    if self.mutation_method == "gaussian":
                        child2 = gaussian_mutation(child2, self.mutation_gene_rate, self.mutation_scale, self.rng)
                    else:
                        child2 = uniform_mutation(child2, self.mutation_gene_rate, self.mutation_scale, self.rng)

                # Append children to next generation if there is space
                if len(next_generation) < self.pop_size:
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
