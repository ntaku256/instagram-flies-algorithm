import numpy as np
from copy import deepcopy

def ACO_Algorithm(problem, **kwargs):
    max_iter = kwargs.get('max_iter', 100)
    pop_size = kwargs.get('pop_size', 100)
    alpha = kwargs.get('alpha', 1.0)  # フェロモンの重要度
    beta = kwargs.get('beta', 2.0)    # ヒューリスティック情報の重要度
    rho = kwargs.get('rho', 0.1)      # フェロモンの蒸発率
    Q = kwargs.get('Q', 1.0)          # フェロモンの増加量
    callback = kwargs.get('callback', None)
    
    # Empty Ant Template
    empty_ant = {
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

    # フェロモン行列の初期化
    pheromone = np.ones((num_var, num_var)) * 0.1
    
    # ヒューリスティック情報の初期化（距離の逆数）
    heuristic = np.ones((num_var, num_var))
    
    # Create Initial Population
    pop = []
    for i in range(pop_size):
        pop.append(deepcopy(empty_ant))
        pop[i]['position'] = np.random.uniform(var_min, var_max, num_var)
        pop[i]['cost'], pop[i]['details'] = cost_function(pop[i]['position'])
        pop[i]['best']['position'] = deepcopy(pop[i]['position'])
        pop[i]['best']['cost'] = pop[i]['cost']
        pop[i]['best']['details'] = pop[i]['details']
        if pop[i]['cost'] < gbest['cost']:
            gbest = deepcopy({
                'position': pop[i]['position'].copy(),
                'cost': pop[i]['cost'],
                'details': pop[i]['details']
            })

    # ACO Loop
    for it in range(max_iter):
        # フェロモンの蒸発
        pheromone *= (1 - rho)
        
        # 各蟻の移動
        for i in range(pop_size):
            # 新しい解の生成
            new_position = np.zeros(num_var)
            for j in range(num_var):
                # 確率の計算
                probs = (pheromone[j] ** alpha) * (heuristic[j] ** beta)
                probs = probs / np.sum(probs)
                
                # 次の位置の選択
                next_pos = np.random.choice(num_var, p=probs)
                new_position[j] = var_min + (var_max - var_min) * next_pos / (num_var - 1)
            
            new_position = np.clip(new_position, var_min, var_max)
            
            # 新しい解の評価
            new_cost, new_details = cost_function(new_position)
            
            # 解の更新
            if new_cost < pop[i]['cost']:
                pop[i]['position'] = new_position
                pop[i]['cost'] = new_cost
                pop[i]['details'] = new_details
                # bestの更新
                if new_cost < pop[i]['best']['cost']:
                    pop[i]['best']['position'] = deepcopy(new_position)
                    pop[i]['best']['cost'] = new_cost
                    pop[i]['best']['details'] = new_details
                
                # フェロモンの更新
                for j in range(num_var):
                    next_pos = int((new_position[j] - var_min) * (num_var - 1) / (var_max - var_min))
                    pheromone[j, next_pos] += Q / new_cost
                
                # グローバルベストの更新
                if new_cost < gbest['cost']:
                    gbest = deepcopy({
                        'position': new_position.copy(),
                        'cost': new_cost,
                        'details': new_details
                    })

        print('Iteration {}: Best Cost = {}'.format(it + 1, gbest['cost']))
        
        if callable(callback):
            callback({
                'it': it + 1,
                'gbest': gbest,
                'pop': pop,
            })

    return gbest, pop 