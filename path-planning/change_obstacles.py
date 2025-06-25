import numpy as np
import path_planning as pp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pso import PSO
from abc_algorithm import ABC_Algorithm
from aco_algorithm import ACO_Algorithm
from instabae_algorithm import IA_Algorithm
from instabae_v2_algorithm import IA_V2_Algorithm
from copy import deepcopy
import os

plt.rcParams["figure.autolayout"] = True

# Create environment
env_params = {
    'width': 100,
    'height': 100,
    'robot_radius': 1,
    'start': [5,5],
    'goal': [95,95],
}
env = pp.Environment(**env_params)

# Obstacles
# obstacles = [
#     {'center': [20, 40], 'radius': 5},
#     {'center': [30, 30], 'radius': 9},
#     {'center': [30, 70], 'radius': 10},
#     {'center': [50, 10], 'radius': 8},
#     {'center': [60, 80], 'radius': 15},
#     {'center': [70, 40], 'radius': 12},
#     {'center': [80, 20], 'radius': 7},
# ]
obstacles = [
  {"center": [9.8, 61.4], "radius": 12},
  {"center": [15.399999999999999, 29.2], "radius": 9},
  {"center": [49.8, 46.800000000000004], "radius": 16},
  {"center": [36.400000000000006, 14.6], "radius": 11},
  {"center": [52.99999999999999, 81.39999999999999], "radius": 13},
  {"center": [70.6, 41.60000000000001], "radius": 7},
  {"center": [74.8, 8.000000000000002], "radius": 16},
  {"center": [91.80277767181397, 63.920834350585935], "radius": 11},
  {"center": [22.40277767181396, 85.32083435058594], "radius": 11},
  {"center": [36.002777671813966, 59.520834350585936], "radius": 4}
]
for obs in obstacles:
    env.add_obstacle(pp.Obstacle(**obs))

# Create cost function
num_control_points = 3
resolution = 50
cost_function = pp.EnvCostFunction(env, num_control_points, resolution)

# Optimization Problem
problem = {
    'num_var': 2*num_control_points,
    'var_min': 0,
    'var_max': 1,
    'cost_function': cost_function,
}

# 保存用リスト
frame_data = []

# callbackでデータだけ貯めておく
def callback(data):
    # if not(data['it'] <= 1 or data['it'] % 2 == 0 or data['it'] == params['max_iter']):
    #     return

    frame_data.append(deepcopy(data))

# アニメーション用の描画
fig, ax = plt.subplots(figsize=[7, 7])
path_line = None
title = ax.set_title("")

def init():
    pp.plot_environment(env)
    global path_line
    first_sol = frame_data[0]['gbest']['details']['sol']
    path_line = pp.plot_path(first_sol, color='b')
    ax.grid(True)
    return path_line, title

def update(frame):
    global path_line
    it = frame['it']
    sol = frame['gbest']['details']['sol']
    length = frame['gbest']['details']['length']

    pp.update_path(sol, path_line)
    title.set_text(f"Iteration: {it}, Length: {length:.2f}")
    return path_line, title

# アルゴリズムの選択
obstacles = 'obstacles2'
algorithm = 'instabae'  # 'pso', 'abc', 'aco', 'instabae' から選択

if algorithm == 'pso':
    # PSO parameters
    params = {
        'max_iter': 100,
        'pop_size': 100,
        'c1': 2,
        'c2': 1,
        'w': 0.8,
        'wdamp': 1,
        'resetting': 25,
    }
    bestsol, pop = PSO(problem, callback=callback, **params)

elif algorithm == 'abc':
    # ABC parameters
    params = {
        'max_iter': 100,
        'pop_size': 100,
        'limit': 50,
    }
    bestsol, pop = ABC_Algorithm(problem, callback=callback, **params)

elif algorithm == 'aco':
    # ACO parameters
    params = {
        'max_iter': 100,
        'pop_size': 100,
        'alpha': 1.0,
        'beta': 2.0,
        'rho': 0.1,
        'Q': 1.0,
    }
    bestsol, pop = ACO_Algorithm(problem, callback=callback, **params)

elif algorithm == 'instabae':
    # Instabae parameters
    params = {
        'max_iter': 100,
        'n_clusters': 10,
        'pop_size': 100,
        'r': 5,
    }
    bestsol, pop = IA_Algorithm(problem, callback=callback, **params)

elif algorithm == 'instabae_v2':
    params = {
        'max_iter': 100,
        'pop_size': 100,
    }
    bestsol, pop = IA_V2_Algorithm(problem, callback=callback, **params)

# なければ保存用ディレクトリを作成
save_dir = f'path-planning/images/{obstacles}'
os.makedirs(save_dir, exist_ok=True)

# アニメーションの作成
ani = animation.FuncAnimation(fig, update, frames=frame_data, init_func=init, blit=False, interval=200, repeat=False)
ani.save(f'path-planning/images/{obstacles}/{algorithm}.gif', writer='pillow')
