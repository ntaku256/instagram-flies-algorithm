from sklearn.cluster import KMeans
import numpy as np
from copy import deepcopy
from sklearn.metrics import silhouette_score

def IA_Algorithm(problem, **kwargs):

    max_iter = kwargs.get('max_iter', 100)
    pop_size = kwargs.get('pop_size', 100)
    initial_clusters = kwargs.get('n_clusters', 10)
    cluster_search_range = kwargs.get('cluster_search_range', (2, 10))
    r = kwargs.get('r', 5)
    callback = kwargs.get('callback', None)

    # Empty Particle Template
    empty_particle = { 
        'position': None,
        'cluster_label': None,
        'strategies': None,
        'cost': None,   
        'details': None,
        'best': {
            'position': None,
            'strategies': None,
            'cost': np.inf,
            'details': None,
        },
    }

    cost_function = problem['cost_function']
    num_var = problem['num_var']
    var_min = problem['var_min']
    var_max = problem['var_max']

    gbest = {
        'position': None,
        'cost': np.inf,
        'details': None,
    }

    centers = None
    center_speeds = None
    center_dist_average = None
    cluster_cost_average = None

    # Create Initial Population
    pop = []
    for i in range(0, pop_size):
        pop.append(deepcopy(empty_particle))
        pop[i]['position'] = np.random.uniform(var_min, var_max, num_var)
        pop[i]['cluster_label'] = np.random.randint(0, initial_clusters)
        pop[i]['strategies'] = np.random.dirichlet([1, 1, 1])
        pop[i]['cost'], pop[i]['details'] = cost_function(pop[i]['position'])
        pop[i]['best']['position'] = deepcopy(pop[i]['position'])
        pop[i]['best']['cost'] = pop[i]['cost']
        pop[i]['best']['details'] = pop[i]['details']
        
        if pop[i]['best']['cost'] < gbest['cost']:
            gbest = deepcopy(pop[i]['best'])

    def pareto(mode, a, shape):
        return (np.random.pareto(a, size=shape) + 1) * mode

    def roulette(weights):
        total = np.sum(weights)
        r = np.random.uniform(0.0, total)
        s = 0
        for i, w in enumerate(weights):
            s += w
            if s > r:
                return i
        return len(weights) - 1

    def clustering():
        nonlocal centers, center_speeds
        positions = np.array([p['position'] for p in pop])
        if centers is None:
            model = KMeans(n_clusters=initial_clusters)
            result = model.fit(positions)
            centers = result.cluster_centers_
        else:
            model = KMeans(n_init=1,n_clusters=initial_clusters,init=centers)
            result = model.fit(positions)
        cluster_labels = result.labels_
        for p, cluster_label in zip(pop, cluster_labels):
            p['cluster_label'] = cluster_label

        center_speeds = result.cluster_centers_ - centers
        centers = result.cluster_centers_

    
        if pop[i]['cost'] < pop[i]['best']['cost']:
            pop[i]['best']['position'] = deepcopy(pop[i]['position'])
            pop[i]['best']['cost'] = pop[i]['cost']
            pop[i]['best']['details'] = pop[i]['details']

    # クラスタのコスト平均の計算
    def update_cluster_cost_average():
        nonlocal cluster_cost_average
        cluster_cost_average = np.zeros(initial_clusters)
        for i in range(initial_clusters):
            cluster_cost_average[i] = np.mean([p['cost'] for p in pop if p['cluster_label'] == i])
        return cluster_cost_average

    # 平均クラスタ間距離の計算
    def update_center_distance():
        nonlocal center_dist_average
        center_dist_average = np.zeros(num_var)
        for i in range(initial_clusters):
            for j in range(i + 1, initial_clusters):
                center_dist_average += centers[i] - centers[j]
        if initial_clusters > 1:
            center_dist_average /= (initial_clusters * (initial_clusters - 1) / 2)

    def Update(strategy, cluster_label, current_pos):
        if strategy == 0:  # pioneer
            return UpdatePioneer(current_pos)   
        elif strategy == 1:  # faddist
            return UpdateFaddist(current_pos)
        elif strategy == 2:  # master
            return UpdateMaster(current_pos, cluster_label)

    def UpdatePioneer(current_pos):
        length = pareto(1, 6, center_dist_average.shape)
        rand01 = np.random.choice([-1, 1], length.shape)
        return current_pos + center_dist_average * length * rand01
    
    def UpdateFaddist(current_pos):
        cluster = roulette(cluster_cost_average)
        return UpdateMaster(current_pos, cluster)

    def UpdateMaster(current_pos, cluster_label):
        candidates = [p for p in pop if p['cluster_label'] == cluster_label]
        if candidates:
            target = candidates[roulette([1.0 / (c['cost'] + 1e-6) for c in candidates])]
            center = centers[cluster_label]
            speed = center_speeds[cluster_label]
            new_pos = current_pos + (center - current_pos) * np.random.rand() \
                        + (target['position'] - current_pos) * np.random.rand() \
                        + speed * np.random.rand()
        else:
            new_pos = current_pos
        return new_pos


    for it in range(max_iter):
        clustering()
        update_cluster_cost_average()
        update_center_distance()

        for i in range(pop_size):
            strategy = roulette(pop[i]['strategies'])
            pos = pop[i]['position']
            cluster_label = pop[i]['cluster_label']

            new_pos = Update(strategy, cluster_label, pos)
            new_pos = np.clip(new_pos, var_min, var_max)
            cost, details = cost_function(new_pos)

            pop[i]['position'] = new_pos
            pop[i]['cost'] = cost
            pop[i]['details'] = details

            if cost < pop[i]['best']['cost']:
                pop[i]['best']['position'] = new_pos
                pop[i]['best']['cost'] = cost
                pop[i]['best']['details'] = details

                if cost < gbest['cost']:
                    gbest = deepcopy(pop[i]['best'])

        print(f"Iteration {it+1}: Best Cost = {gbest['cost']}")

        if callable(callback):
            callback({
                'it': it + 1,
                'gbest': deepcopy(gbest),
                'pop': deepcopy(pop),
                'centers': deepcopy(centers)
            })

    return gbest, pop
