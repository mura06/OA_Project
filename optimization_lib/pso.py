import numpy as np
from optimization_lib.network import fitness_function
from optimization_lib.ga import init_uniform, init_normal

class ParticleSwarmOptimization:
    def __init__(self, model, X, y, num_particles=50, generations=100,
                 w_start=0.9, w_end=0.4, c1=1.5, c2=1.5, v_max=0.5,
                 pos_min=-2.0, pos_max=2.0, init_method="uniform",
                 metric="f1_macro", random_state=42):
        self.model = model
        self.X = X
        self.y = y
        self.num_particles = num_particles
        self.generations = generations
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        self.pos_min = pos_min
        self.pos_max = pos_max
        self.init_method = init_method
        self.metric = metric
        self.rng = np.random.default_rng(random_state)
        
        # Determine number of network parameters
        dummy_weights = []
        for coef, intercept in zip(model.coefs_, model.intercepts_):
            dummy_weights.append(coef.flatten())
            dummy_weights.append(intercept.flatten())
        self.num_params = len(np.concatenate(dummy_weights))

    def solve(self):
        # 1. Initialize particle positions and velocities
        positions = []
        velocities = []
        pbest_positions = []
        pbest_fitnesses = []
        
        for _ in range(self.num_particles):
            # Position initialization
            if self.init_method == "uniform":
                pos = init_uniform(self.model, self.rng)
            else:
                pos = init_normal(self.model, self.rng)
            
            # Velocity initialization (small random values)
            vel = self.rng.uniform(-self.v_max * 0.5, self.v_max * 0.5, size=self.num_params)
            
            positions.append(pos)
            velocities.append(vel)
            pbest_positions.append(pos.copy())
            
            # Compute initial fitness
            fit = fitness_function(pos, self.model, self.X, self.y, self.metric)
            pbest_fitnesses.append(fit)

        positions = np.array(positions)
        velocities = np.array(velocities)
        pbest_positions = np.array(pbest_positions)
        pbest_fitnesses = np.array(pbest_fitnesses)

        # Swarm best tracking
        gbest_idx = np.argmax(pbest_fitnesses)
        gbest_fitness = pbest_fitnesses[gbest_idx]
        gbest_position = pbest_positions[gbest_idx].copy()

        gbest_fitness_history = []
        mean_fitness_history = []

        # 2. Optimization loop
        for gen in range(self.generations):
            # Update inertia weight w (linear decay)
            w = self.w_start - (self.w_start - self.w_end) * (gen / self.generations)

            for i in range(self.num_particles):
                # Random coefficients
                r1 = self.rng.uniform(0, 1, size=self.num_params)
                r2 = self.rng.uniform(0, 1, size=self.num_params)

                # Cognitive and social velocity components
                cognitive = self.c1 * r1 * (pbest_positions[i] - positions[i])
                social = self.c2 * r2 * (gbest_position - positions[i])

                # Velocity update
                velocities[i] = w * velocities[i] + cognitive + social

                # Velocity clamping
                velocities[i] = np.clip(velocities[i], -self.v_max, self.v_max)

                # Position update
                positions[i] = positions[i] + velocities[i]

                # Position boundary clamping (numerical stability)
                positions[i] = np.clip(positions[i], self.pos_min, self.pos_max)

                # Evaluate fitness
                fit = fitness_function(positions[i], self.model, self.X, self.y, self.metric)

                # Update personal best (pbest)
                if fit > pbest_fitnesses[i]:
                    pbest_fitnesses[i] = fit
                    pbest_positions[i] = positions[i].copy()

                    # Update global best (gbest)
                    if fit > gbest_fitness:
                        gbest_fitness = fit
                        gbest_position = positions[i].copy()

            gbest_fitness_history.append(gbest_fitness)
            mean_fitness_history.append(np.mean(pbest_fitnesses))

        return gbest_position, gbest_fitness, gbest_fitness_history, mean_fitness_history
