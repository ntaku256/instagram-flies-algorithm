import numpy as np
import pandas as pd
from pso import PSO
from abc_algorithm import ABC_Algorithm
from aco_algorithm import ACO_Algorithm
from instabae_algorithm import IA_Algorithm
from ga_algorithm import GA_Algorithm
import path_planning as pp

# パラメータ
algorithm = 'abc'  # 'pso', 'abc', 'aco', 'instabae', 'ga'
n_trials = 100
max_iter = 100

# 問題定義
env_params = {
    'width': 100,
    'height': 100,
    'robot_radius': 1,
    'start': [5,5],
    'goal': [95,95],
}
env = pp.Environment(**env_params)
obstacles = [
    {'center': [20, 40], 'radius': 5},
    {'center': [30, 30], 'radius': 9},
    {'center': [30, 70], 'radius': 10},
    {'center': [50, 10], 'radius': 8},
    {'center': [60, 80], 'radius': 15},
    {'center': [70, 40], 'radius': 12},
    {'center': [80, 20], 'radius': 7},
]
for obs in obstacles:
    env.add_obstacle(pp.Obstacle(**obs))
num_control_points = 3
resolution = 50
cost_function = pp.EnvCostFunction(env, num_control_points, resolution)
problem = {
    'num_var': 2*num_control_points,
    'var_min': 0,
    'var_max': 1,
    'cost_function': cost_function,
}

# アルゴリズムごとのパラメータ
template_params = {
    'pso': dict(max_iter=max_iter, pop_size=100, c1=2, c2=1, w=0.8, wdamp=1, resetting=25),
    'abc': dict(max_iter=max_iter, pop_size=100, limit=50),
    'aco': dict(max_iter=max_iter, pop_size=100, alpha=1.0, beta=2.0, rho=0.1, Q=1.0),
    'instabae': dict(max_iter=max_iter, n_clusters=10, pop_size=100, r=5),
    'ga': dict(max_iter=max_iter, pop_size=100, crossover_rate=0.8, mutation_rate=0.1, mutation_step=0.1, 
               selection_method='tournament', crossover_method='uniform', tournament_size=3, elite_count=2),
}

algo_params = template_params[algorithm]

# gbest推移を保存
all_gbest_histories = []

for trial in range(n_trials):
    gbest_history = []
    def callback(data):
        gbest_history.append(data['gbest']['cost'])
    if algorithm == 'pso':
        PSO(problem, callback=callback, **algo_params)
    elif algorithm == 'abc':
        ABC_Algorithm(problem, callback=callback, **algo_params)
    elif algorithm == 'aco':
        ACO_Algorithm(problem, callback=callback, **algo_params)
    elif algorithm == 'instabae':
        IA_Algorithm(problem, callback=callback, **algo_params)
    elif algorithm == 'ga':
        GA_Algorithm(problem, callback=callback, **algo_params)
    all_gbest_histories.append(gbest_history)

# DataFrame化
max_len = min(len(hist) for hist in all_gbest_histories)  # 最短に揃える
all_gbest_histories = [hist[:max_len] for hist in all_gbest_histories]
df = pd.DataFrame(np.array(all_gbest_histories).T, columns=[f'Trial{i+1}' for i in range(n_trials)])

# 平均と標準偏差を左側に追加
df.insert(0, 'Average', df.mean(axis=1))
df.insert(1, 'Std', df.std(axis=1))

# Excel出力
df.to_excel(f'path-planning/excel/gbest/{algorithm}.xlsx', index_label='Iteration')

