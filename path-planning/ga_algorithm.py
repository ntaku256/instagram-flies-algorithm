import numpy as np
from copy import deepcopy

# Genetic Algorithm
def GA_Algorithm(problem, **kwargs):

    max_iter = kwargs.get('max_iter', 100)
    pop_size = kwargs.get('pop_size', 100)
    crossover_rate = kwargs.get('crossover_rate', 0.8)
    mutation_rate = kwargs.get('mutation_rate', 0.1)
    mutation_step = kwargs.get('mutation_step', 0.1)
    selection_method = kwargs.get('selection_method', 'tournament')
    crossover_method = kwargs.get('crossover_method', 'uniform')
    tournament_size = kwargs.get('tournament_size', 3)
    elite_count = kwargs.get('elite_count', 2)
    callback = kwargs.get('callback', None)
    
    # Empty Individual Template
    empty_individual = {
        'position': None,
        'cost': None,
        'details': None,
        'best': {
            'position': None,
            'cost': np.inf,
            'details': None,
        },
    }

    # Extract Problem Info
    cost_function = problem['cost_function']
    var_min = problem['var_min']
    var_max = problem['var_max']
    num_var = problem['num_var']
    
    # Initialize Global Best
    gbest = {
        'position': None,
        'cost': np.inf,
        'details': None,
    }

    # Create Initial Population
    pop = []
    for i in range(0, pop_size):
        pop.append(deepcopy(empty_individual))
        pop[i]['position'] = np.random.uniform(var_min, var_max, num_var)
        pop[i]['cost'], pop[i]['details'] = cost_function(pop[i]['position'])
        pop[i]['best']['position'] = deepcopy(pop[i]['position'])
        pop[i]['best']['cost'] = pop[i]['cost']
        pop[i]['best']['details'] = pop[i]['details']
        
        if pop[i]['cost'] < gbest['cost']:
            gbest = deepcopy(pop[i])
    
    # Selection Functions
    def tournament_selection(population, tournament_size):
        selected = []
        for _ in range(len(population)):
            tournament = np.random.choice(len(population), tournament_size, replace=False)
            winner = min(tournament, key=lambda x: population[x]['cost'])
            selected.append(deepcopy(population[winner]))
        return selected
    
    def roulette_wheel_selection(population):
        # Convert costs to fitness (lower cost = higher fitness)
        costs = np.array([ind['cost'] for ind in population])
        max_cost = np.max(costs)
        fitness = max_cost - costs + 1e-10  # Add small value to avoid division by zero
        
        # Calculate selection probabilities
        probabilities = fitness / np.sum(fitness)
        
        selected = []
        for _ in range(len(population)):
            idx = np.random.choice(len(population), p=probabilities)
            selected.append(deepcopy(population[idx]))
        return selected
    
    # Crossover Functions
    def uniform_crossover(parent1, parent2):
        child1 = deepcopy(parent1)
        child2 = deepcopy(parent2)
        
        mask = np.random.rand(num_var) < 0.5
        child1['position'][mask] = parent2['position'][mask]
        child2['position'][~mask] = parent1['position'][~mask]
        
        return child1, child2
    
    def single_point_crossover(parent1, parent2):
        child1 = deepcopy(parent1)
        child2 = deepcopy(parent2)
        
        point = np.random.randint(1, num_var)
        child1['position'][point:] = parent2['position'][point:]
        child2['position'][point:] = parent1['position'][point:]
        
        return child1, child2
    
    def arithmetic_crossover(parent1, parent2):
        alpha = np.random.rand()
        child1 = deepcopy(parent1)
        child2 = deepcopy(parent2)
        
        child1['position'] = alpha * parent1['position'] + (1 - alpha) * parent2['position']
        child2['position'] = (1 - alpha) * parent1['position'] + alpha * parent2['position']
        
        return child1, child2
    
    # Mutation Function
    def mutate(individual):
        mutated = deepcopy(individual)
        mask = np.random.rand(num_var) < mutation_rate
        
        # Gaussian mutation
        mutation_values = np.random.normal(0, mutation_step, num_var)
        mutated['position'][mask] += mutation_values[mask]
        
        # Ensure bounds
        mutated['position'] = np.clip(mutated['position'], var_min, var_max)
        
        return mutated
    
    # GA Main Loop
    for it in range(0, max_iter):
        
        # Selection
        if selection_method == 'tournament':
            selected_pop = tournament_selection(pop, tournament_size)
        elif selection_method == 'roulette':
            selected_pop = roulette_wheel_selection(pop)
        else:
            selected_pop = tournament_selection(pop, tournament_size)  # Default
        
        # Create new population
        new_pop = []
        
        # Elite preservation
        pop_sorted = sorted(pop, key=lambda x: x['cost'])
        for i in range(elite_count):
            new_pop.append(deepcopy(pop_sorted[i]))
        
        # Crossover and Mutation
        while len(new_pop) < pop_size:
            # Select parents
            parent1 = selected_pop[np.random.randint(len(selected_pop))]
            parent2 = selected_pop[np.random.randint(len(selected_pop))]
            
            # Crossover
            if np.random.rand() < crossover_rate:
                if crossover_method == 'uniform':
                    child1, child2 = uniform_crossover(parent1, parent2)
                elif crossover_method == 'single_point':
                    child1, child2 = single_point_crossover(parent1, parent2)
                elif crossover_method == 'arithmetic':
                    child1, child2 = arithmetic_crossover(parent1, parent2)
                else:
                    child1, child2 = uniform_crossover(parent1, parent2)  # Default
            else:
                child1, child2 = deepcopy(parent1), deepcopy(parent2)
            
            # Mutation
            child1 = mutate(child1)
            child2 = mutate(child2)
            
            new_pop.extend([child1, child2])
        
        # Trim population to exact size
        new_pop = new_pop[:pop_size]
        
        # Evaluate new population and update best for each individual
        for i in range(len(new_pop)):
            new_pop[i]['cost'], new_pop[i]['details'] = cost_function(new_pop[i]['position'])
            
            # Initialize best if not set (for new offspring)
            if new_pop[i]['best']['position'] is None:
                new_pop[i]['best']['position'] = deepcopy(new_pop[i]['position'])
                new_pop[i]['best']['cost'] = new_pop[i]['cost']
                new_pop[i]['best']['details'] = new_pop[i]['details']
            # Update best if current is better
            elif new_pop[i]['cost'] < new_pop[i]['best']['cost']:
                new_pop[i]['best']['position'] = deepcopy(new_pop[i]['position'])
                new_pop[i]['best']['cost'] = new_pop[i]['cost']
                new_pop[i]['best']['details'] = new_pop[i]['details']
            
            if new_pop[i]['cost'] < gbest['cost']:
                gbest = deepcopy(new_pop[i])
        
        # Replace population
        pop = new_pop
        
        print('Iteration {}: Best Cost = {}'.format(it + 1, gbest['cost']))
        
        if callable(callback):
            callback({
                'it': it + 1,
                'gbest': gbest,
                'pop': pop,
            })

    return gbest, pop 
