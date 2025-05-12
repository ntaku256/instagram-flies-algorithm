import numpy as np
from copy import deepcopy

def ABC(problem, **kwargs):
    max_iter = kwargs.get('max_iter', 100)
    pop_size = kwargs.get('pop_size', 100)
    limit = kwargs.get('limit', 50)  # 試行回数の制限
    callback = kwargs.get('callback', None)
    
    # Empty Bee Template
    empty_bee = {
        'position': None,
        'cost': None,
        'details': None,
        'trial': 0,  # 試行回数
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
    for i in range(pop_size):
        pop.append(deepcopy(empty_bee))
        pop[i]['position'] = np.random.uniform(var_min, var_max, num_var)
        pop[i]['cost'], pop[i]['details'] = cost_function(pop[i]['position'])
        
        if pop[i]['cost'] < gbest['cost']:
            gbest = deepcopy({
                'position': pop[i]['position'].copy(),
                'cost': pop[i]['cost'],
                'details': pop[i]['details']
            })

    # ABC Loop
    for it in range(max_iter):
        # Employed Bees Phase
        for i in range(pop_size):
            # 新しい解の生成
            phi = np.random.uniform(-1, 1, num_var)
            j = np.random.randint(0, pop_size)
            new_position = pop[i]['position'] + phi * (pop[i]['position'] - pop[j]['position'])
            new_position = np.clip(new_position, var_min, var_max)
            
            # 新しい解の評価
            new_cost, new_details = cost_function(new_position)
            
            # 貪欲選択
            if new_cost < pop[i]['cost']:
                pop[i]['position'] = new_position
                pop[i]['cost'] = new_cost
                pop[i]['details'] = new_details
                pop[i]['trial'] = 0
            else:
                pop[i]['trial'] += 1

            # グローバルベストの更新
            if pop[i]['cost'] < gbest['cost']:
                gbest = deepcopy({
                    'position': pop[i]['position'].copy(),
                    'cost': pop[i]['cost'],
                    'details': pop[i]['details']
                })

        # Onlooker Bees Phase
        fitness = np.array([1.0 / (1.0 + bee['cost']) for bee in pop])
        fitness = fitness / np.sum(fitness)
        
        for i in range(pop_size):
            # ルーレット選択
            selected = np.random.choice(pop_size, p=fitness)
            phi = np.random.uniform(-1, 1, num_var)
            j = np.random.randint(0, pop_size)
            new_position = pop[selected]['position'] + phi * (pop[selected]['position'] - pop[j]['position'])
            new_position = np.clip(new_position, var_min, var_max)
            
            # 新しい解の評価
            new_cost, new_details = cost_function(new_position)
            
            # 貪欲選択
            if new_cost < pop[selected]['cost']:
                pop[selected]['position'] = new_position
                pop[selected]['cost'] = new_cost
                pop[selected]['details'] = new_details
                pop[selected]['trial'] = 0
            else:
                pop[selected]['trial'] += 1

            # グローバルベストの更新
            if pop[selected]['cost'] < gbest['cost']:
                gbest = deepcopy({
                    'position': pop[selected]['position'].copy(),
                    'cost': pop[selected]['cost'],
                    'details': pop[selected]['details']
                })

        # Scout Bees Phase
        for i in range(pop_size):
            if pop[i]['trial'] >= limit:
                pop[i]['position'] = np.random.uniform(var_min, var_max, num_var)
                pop[i]['cost'], pop[i]['details'] = cost_function(pop[i]['position'])
                pop[i]['trial'] = 0

        print('Iteration {}: Best Cost = {}'.format(it + 1, gbest['cost']))
        
        if callable(callback):
            callback({
                'it': it + 1,
                'gbest': gbest,
                'pop': pop,
            })

    return gbest, pop 