import numpy as np
from copy import deepcopy

def ABC_Algorithm(problem, **kwargs):
    max_iter = kwargs.get('max_iter', 100)
    pop_size = kwargs.get('pop_size', 100)
    limit = kwargs.get('limit', 50)
    callback = kwargs.get('callback', None)
    
    empty_bee = {
        'position': None,
        'cost': None,
        'details': None,
        'trial': 0,
        'best': {
            'position': None,
            'cost': np.inf,
            'details': None,
        },
    }

    cost_function = problem['cost_function']
    var_min = problem['var_min']
    var_max = problem['var_max']
    num_var = problem['num_var'] * 2

    gbest = {
        'position': None,
        'cost': np.inf,
        'details': None,
    }

    pop = []
    for i in range(pop_size):
        pop.append(deepcopy(empty_bee))
        pos = np.random.uniform(var_min, var_max, num_var)
        cost, details = cost_function(pos)
        pop[i]['position'] = pos
        pop[i]['cost'] = cost
        pop[i]['details'] = details
        pop[i]['best']['position'] = deepcopy(pos)
        pop[i]['best']['cost'] = cost
        pop[i]['best']['details'] = details
        if cost < gbest['cost']:
            gbest = deepcopy({'position': pos.copy(), 'cost': cost, 'details': details})

    for it in range(max_iter):
        for i in range(pop_size):
            j = np.random.randint(0, pop_size)
            if j == i:
                j = (j + 1) % pop_size
            phi = np.random.uniform(-1, 1, num_var) * 0.1  # 小さく
            # 改良: gbest方向の探索
            if np.random.rand() < 0.5:
                new_position = pop[i]['position'] + phi * (gbest['position'] - pop[i]['position'])
            else:
                new_position = pop[i]['position'] + phi * (pop[i]['position'] - pop[j]['position'])

            new_position = np.clip(new_position, var_min, var_max)
            new_cost, new_details = cost_function(new_position)

            if new_cost < pop[i]['cost']:
                pop[i]['position'] = new_position
                pop[i]['cost'] = new_cost
                pop[i]['details'] = new_details
                pop[i]['trial'] = 0
                if new_cost < pop[i]['best']['cost']:
                    pop[i]['best']['position'] = deepcopy(new_position)
                    pop[i]['best']['cost'] = new_cost
                    pop[i]['best']['details'] = new_details
            else:
                pop[i]['trial'] += 1

            if new_cost < gbest['cost']:
                gbest = deepcopy({'position': new_position.copy(), 'cost': new_cost, 'details': new_details})

        # ソフトマックスによる確率選択
        raw_fitness = np.array([bee['cost'] for bee in pop])
        fitness = np.exp(-raw_fitness / (np.std(raw_fitness) + 1e-8))
        fitness = fitness / np.sum(fitness)

        for _ in range(pop_size):
            selected = np.random.choice(pop_size, p=fitness)
            j = np.random.randint(0, pop_size)
            if j == selected:
                j = (j + 1) % pop_size
            phi = np.random.uniform(-1, 1, num_var) * 0.1
            if np.random.rand() < 0.5:
                new_position = pop[selected]['position'] + phi * (gbest['position'] - pop[selected]['position'])
            else:
                new_position = pop[selected]['position'] + phi * (pop[selected]['position'] - pop[j]['position'])

            new_position = np.clip(new_position, var_min, var_max)
            new_cost, new_details = cost_function(new_position)

            if new_cost < pop[selected]['cost']:
                pop[selected]['position'] = new_position
                pop[selected]['cost'] = new_cost
                pop[selected]['details'] = new_details
                pop[selected]['trial'] = 0
                if new_cost < pop[selected]['best']['cost']:
                    pop[selected]['best']['position'] = deepcopy(new_position)
                    pop[selected]['best']['cost'] = new_cost
                    pop[selected]['best']['details'] = new_details
            else:
                pop[selected]['trial'] += 1

            if new_cost < gbest['cost']:
                gbest = deepcopy({'position': new_position.copy(), 'cost': new_cost, 'details': new_details})

        for i in range(pop_size):
            if pop[i]['trial'] >= limit:
                # 改良: gbest近傍に再配置
                noise = np.random.normal(0, 0.1, num_var) * (var_max - var_min)
                new_pos = gbest['position'] + noise
                new_pos = np.clip(new_pos, var_min, var_max)
                cost, details = cost_function(new_pos)
                pop[i]['position'] = new_pos
                pop[i]['cost'] = cost
                pop[i]['details'] = details
                pop[i]['trial'] = 0
                pop[i]['best']['position'] = deepcopy(new_pos)
                pop[i]['best']['cost'] = cost
                pop[i]['best']['details'] = details
                if cost < gbest['cost']:
                    gbest = deepcopy({'position': new_pos.copy(), 'cost': cost, 'details': details})

        print(f'Iteration {it + 1}: Best Cost = {gbest["cost"]:.6f}')

        if callable(callback):
            callback({
                'it': it + 1,
                'gbest': gbest,
                'pop': pop,
            })

    return gbest, pop
