import numpy as np
from copy import deepcopy

# Cooperatively Coevolving Particle Swarm Optimization (CPSO)
def CPSO(problem, **kwargs):

    max_iter = kwargs.get('max_iter', 100)
    pop_size = kwargs.get('pop_size', 100)
    n_subs = kwargs.get('n_subs', 2)  # サブスワームの数
    c1 = kwargs.get('c1', 1.4962)
    c2 = kwargs.get('c2', 1.4962)
    w = kwargs.get('w', 0.7298)
    wdamp = kwargs.get('wdamp', 1.0)
    callback = kwargs.get('callback', None)
    resetting = kwargs.get('resetting', None)
    
    # Extract Problem Info
    cost_function = problem['cost_function']
    var_min = problem['var_min']
    var_max = problem['var_max']
    num_var = problem['num_var']
    
    # サブコンポーネントへの分割
    dims_per_sub = num_var // n_subs
    remainder = num_var % n_subs
    
    sub_dims = []
    sub_indices = []
    start_idx = 0
    for i in range(n_subs):
        # 余りを最初のサブスワームに分配
        sub_dim = dims_per_sub + (1 if i < remainder else 0)
        sub_dims.append(sub_dim)
        sub_indices.append((start_idx, start_idx + sub_dim))
        start_idx += sub_dim
    
    # Empty Particle Template
    empty_particle = {
        'position': None,
        'velocity': None,
        'cost': None,
        'details': None,
        'best': {
            'position': None,
            'cost': np.inf,
            'details': None,
        },
    }
    
    # 各サブスワームの初期化
    sub_pops = []  # 各サブスワームの粒子群
    sub_gbests = []  # 各サブスワームの群最良解
    
    for sub_idx in range(n_subs):
        sub_dim = sub_dims[sub_idx]
        sub_var_min = var_min if np.isscalar(var_min) else var_min[sub_indices[sub_idx][0]:sub_indices[sub_idx][1]]
        sub_var_max = var_max if np.isscalar(var_max) else var_max[sub_indices[sub_idx][0]:sub_indices[sub_idx][1]]
        
        # このサブスワームの粒子群
        sub_pop = []
        sub_gbest = {
            'position': None,
            'cost': np.inf,
            'details': None,
        }
        
        for i in range(pop_size):
            particle = deepcopy(empty_particle)
            particle['position'] = np.random.uniform(sub_var_min, sub_var_max, sub_dim)
            particle['velocity'] = np.zeros(sub_dim)
            sub_pop.append(particle)
        
        sub_pops.append(sub_pop)
        sub_gbests.append(sub_gbest)
    
    # Initialize Global Best
    gbest = {
        'position': None,
        'cost': np.inf,
        'details': None,
    }
    
    def _construct_complete_vector(sub_idx, particle_idx):
        """
        指定されたサブスワームの粒子と他のサブスワームの最良解を組み合わせて
        完全な解ベクトルを構築
        """
        complete_vector = np.zeros(num_var)
        
        for i in range(n_subs):
            start_idx, end_idx = sub_indices[i]
            if i == sub_idx:
                # 対象のサブスワームの粒子を使用
                complete_vector[start_idx:end_idx] = sub_pops[i][particle_idx]['position']
            else:
                # 他のサブスワームの最良解を使用
                if sub_gbests[i]['position'] is not None:
                    complete_vector[start_idx:end_idx] = sub_gbests[i]['position']
                else:
                    # 初期化されていない場合はランダムな値を使用
                    sub_var_min = var_min if np.isscalar(var_min) else var_min[start_idx:end_idx]
                    sub_var_max = var_max if np.isscalar(var_max) else var_max[start_idx:end_idx]
                    complete_vector[start_idx:end_idx] = np.random.uniform(sub_var_min, sub_var_max)
        
        return complete_vector
    
    def _evaluate_sub_swarm(sub_idx):
        """指定されたサブスワームの全ての粒子を評価"""
        for particle_idx in range(pop_size):
            # 完全な解ベクトルを構築
            complete_vector = _construct_complete_vector(sub_idx, particle_idx)
            
            # 評価
            cost, details = cost_function(complete_vector)
            
            # 個体最良解の更新
            if cost < sub_pops[sub_idx][particle_idx]['best']['cost']:
                sub_pops[sub_idx][particle_idx]['best']['position'] = deepcopy(
                    sub_pops[sub_idx][particle_idx]['position'])
                sub_pops[sub_idx][particle_idx]['best']['cost'] = cost
                sub_pops[sub_idx][particle_idx]['best']['details'] = details
            
            # サブスワームの群最良解の更新
            if cost < sub_gbests[sub_idx]['cost']:
                sub_gbests[sub_idx]['position'] = deepcopy(
                    sub_pops[sub_idx][particle_idx]['position'])
                sub_gbests[sub_idx]['cost'] = cost
                sub_gbests[sub_idx]['details'] = details
            
            # 全体の最良解の更新
            if cost < gbest['cost']:
                gbest['position'] = deepcopy(complete_vector)
                gbest['cost'] = cost
                gbest['details'] = details
            
            # 現在のコストを保存
            sub_pops[sub_idx][particle_idx]['cost'] = cost
            sub_pops[sub_idx][particle_idx]['details'] = details
    
    # 初期評価
    for sub_idx in range(n_subs):
        _evaluate_sub_swarm(sub_idx)
    
    # CPSO Loop
    for it in range(0, max_iter):
        do_resetting = resetting and ((it + 1) % resetting == 0)
        if do_resetting:
            print('Resetting particles...')
        
        # 各サブスワームを順番に更新
        for sub_idx in range(n_subs):
            sub_dim = sub_dims[sub_idx]
            sub_var_min = var_min if np.isscalar(var_min) else var_min[sub_indices[sub_idx][0]:sub_indices[sub_idx][1]]
            sub_var_max = var_max if np.isscalar(var_max) else var_max[sub_indices[sub_idx][0]:sub_indices[sub_idx][1]]
            
            for i in range(pop_size):
                if do_resetting:
                    sub_pops[sub_idx][i]['position'] = np.random.uniform(sub_var_min, sub_var_max, sub_dim)
                    sub_pops[sub_idx][i]['velocity'] = np.zeros(sub_dim)
                else:
                    # PSOの速度更新式
                    r1 = np.random.rand(sub_dim)
                    r2 = np.random.rand(sub_dim)
                    
                    sub_pops[sub_idx][i]['velocity'] = w * sub_pops[sub_idx][i]['velocity'] \
                        + c1 * r1 * (sub_pops[sub_idx][i]['best']['position'] - sub_pops[sub_idx][i]['position']) \
                        + c2 * r2 * (sub_gbests[sub_idx]['position'] - sub_pops[sub_idx][i]['position'])
                    
                    # 位置更新
                    sub_pops[sub_idx][i]['position'] += sub_pops[sub_idx][i]['velocity']
                    sub_pops[sub_idx][i]['position'] = np.clip(sub_pops[sub_idx][i]['position'], 
                                                               sub_var_min, sub_var_max)
            
            # このサブスワームを評価
            _evaluate_sub_swarm(sub_idx)
        
        w *= wdamp
        print('Iteration {}: Best Cost = {}'.format(it + 1, gbest['cost']))
        
        if callable(callback):
            # 完全な粒子群を構築（コールバック用）
            complete_pop = []
            for i in range(pop_size):
                # 各サブスワームから対応するインデックスの粒子を選んで完全な解を構築
                complete_vector = np.zeros(num_var)
                complete_cost = 0
                complete_details = None
                
                for sub_idx in range(n_subs):
                    start_idx, end_idx = sub_indices[sub_idx]
                    complete_vector[start_idx:end_idx] = sub_pops[sub_idx][i]['best']['position']
                
                # 完全な解を評価
                complete_cost, complete_details = cost_function(complete_vector)
                
                complete_particle = deepcopy(empty_particle)
                complete_particle['position'] = complete_vector
                complete_particle['cost'] = complete_cost
                complete_particle['details'] = complete_details
                complete_particle['best']['position'] = deepcopy(complete_vector)
                complete_particle['best']['cost'] = complete_cost
                complete_particle['best']['details'] = complete_details
                complete_particle['velocity'] = np.zeros(num_var)  # 完全な解の速度は0
                complete_pop.append(complete_particle)
            
            callback({
                'it': it + 1,
                'gbest': deepcopy(gbest),
                'pop': complete_pop,
            })
    
    # 最終的な完全な粒子群を構築（返り値用）
    complete_pop = []
    for i in range(pop_size):
        # 各サブスワームから対応するインデックスの粒子を選んで完全な解を構築
        complete_vector = np.zeros(num_var)
        
        for sub_idx in range(n_subs):
            start_idx, end_idx = sub_indices[sub_idx]
            complete_vector[start_idx:end_idx] = sub_pops[sub_idx][i]['best']['position']
        
        # 完全な解を評価
        complete_cost, complete_details = cost_function(complete_vector)
        
        complete_particle = deepcopy(empty_particle)
        complete_particle['position'] = complete_vector
        complete_particle['cost'] = complete_cost
        complete_particle['details'] = complete_details
        complete_particle['best']['position'] = deepcopy(complete_vector)
        complete_particle['best']['cost'] = complete_cost
        complete_particle['best']['details'] = complete_details
        complete_particle['velocity'] = np.zeros(num_var)
        complete_pop.append(complete_particle)
    
    return gbest, complete_pop

