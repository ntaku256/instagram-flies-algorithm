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
from ga_algorithm import GA_Algorithm
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
  {"center": [21.6, 3.3625000000000016], "radius": 4},
  {"center": [22.8, 10.3625], "radius": 4},
  {"center": [21.8, 18.3625], "radius": 4},
  {"center": [19.8, 26.5625], "radius": 4},
  {"center": [25, 23.5625], "radius": 4},
  {"center": [23.2, 35.1625], "radius": 4},
  {"center": [24.8, 27.3625], "radius": 4},
  {"center": [22.4, 44.7625], "radius": 4},
  {"center": [28, 39.9625], "radius": 4},
  {"center": [22, 40.7625], "radius": 4},
  {"center": [19.6, 47.7625], "radius": 4},
  {"center": [20, 53.7625], "radius": 4},
  {"center": [21.4, 61.3625], "radius": 4},
  {"center": [25, 56.5625], "radius": 4},
  {"center": [30.8, 49.3625], "radius": 4},
  {"center": [27.4, 17.3625], "radius": 4},
  {"center": [26.599999999999998, 7.1625000000000005], "radius": 4},
  {"center": [26.8, 13.1625], "radius": 4},
  {"center": [21.4, 67.3625], "radius": 4},
  {"center": [20.4, 74.5625], "radius": 4},
  {"center": [20.8, 79.3625], "radius": 4},
  {"center": [58.2, 99.1625], "radius": 4},
  {"center": [59.4, 92.3625], "radius": 4},
  {"center": [59.4, 87.9625], "radius": 4},
  {"center": [58.6, 82.9625], "radius": 4},
  {"center": [59.00000000000001, 75.76249999999999], "radius": 4},
  {"center": [59.4, 66.3625], "radius": 4},
  {"center": [63.6, 71.7625], "radius": 4},
  {"center": [60.8, 61.9625], "radius": 4},
  {"center": [59.800000000000004, 52.5625], "radius": 4},
  {"center": [60.6, 43.1625], "radius": 4},
  {"center": [63.8, 57.1625], "radius": 4},
  {"center": [61.8, 48.9625], "radius": 4},
  {"center": [61.199999999999996, 37.762499999999996], "radius": 4},
  {"center": [59.8, 30.3625], "radius": 4},
  {"center": [59.2, 23.7625], "radius": 4},
  {"center": [59.2, 17.7625], "radius": 4},
  {"center": [65.2, 17.7625], "radius": 4},
  {"center": [64.2, 22.3625], "radius": 4},
  {"center": [64.2, 32.3625], "radius": 4},
  {"center": [68.2, 27.9625], "radius": 4},
  {"center": [65.8, 39.9625], "radius": 4},
  {"center": [66.2, 44.3625], "radius": 4},
  {"center": [67.6, 52.3625], "radius": 4},
  {"center": [68.4, 57.7625], "radius": 4},
  {"center": [67.4, 63.7625], "radius": 4},
  {"center": [69.4, 69.9625], "radius": 4},
  {"center": [67.8, 79.1625], "radius": 4},
  {"center": [72.6, 75.1625], "radius": 4},
  {"center": [63.8, 81.1625], "radius": 4},
  {"center": [64.4, 88.16250000000001], "radius": 4},
  {"center": [65.4, 93.3625], "radius": 4},
  {"center": [63, 97.9625], "radius": 4},
  {"center": [26.6, 78.1625], "radius": 4},
  {"center": [25.4, 72.9625], "radius": 4},
  {"center": [26, 66.7625], "radius": 4},
  {"center": [26.2, 62.3625], "radius": 4},
  {"center": [29.8, 59.3625], "radius": 4},
  {"center": [29.4, 54.5625], "radius": 4},
  {"center": [25, 51.9625], "radius": 4},
  {"center": [27, 45.3625], "radius": 4},
  {"center": [29, 35.1625], "radius": 4},
  {"center": [29.6, 30.1625], "radius": 4},
  {"center": [29, 25.5625], "radius": 4},
  {"center": [30.6, 14.7625], "radius": 4},
  {"center": [29.8, 9.9625], "radius": 4},
  {"center": [28, 2.3625], "radius": 4},
  {"center": [70.8, 17.9625], "radius": 4},
  {"center": [76.2, 18.9625], "radius": 4},
  {"center": [83.8, 18.5625], "radius": 4},
  {"center": [82.4, 25.1625], "radius": 4},
  {"center": [74.2, 24.3625], "radius": 4},
  {"center": [34.2, 77.9625], "radius": 4},
  {"center": [32, 73.9625], "radius": 4},
  {"center": [55.4, 27.1625], "radius": 4},
  {"center": [33.4, 69.1625], "radius": 4},
  {"center": [70.6, 98.1625], "radius": 4},
  {"center": [71.4, 94.1625], "radius": 4},
  {"center": [70.2, 89.1625], "radius": 4},
  {"center": [71.2, 84.3625], "radius": 4},
  {"center": [73.4, 80.7625], "radius": 4},
  {"center": [64, 76.5625], "radius": 4},
  {"center": [68.2, 74.1625], "radius": 4},
  {"center": [64.2, 66.7625], "radius": 4},
  {"center": [59.2, 70.9625], "radius": 4},
  {"center": [29.2, 70.3625], "radius": 4},
  {"center": [33, 2.1625], "radius": 4},
  {"center": [33.4, 7.1625], "radius": 4},
  {"center": [15.4, 0.9625], "radius": 4},
  {"center": [17, 7.5625], "radius": 4},
  {"center": [17.6, 12.9625], "radius": 4},
  {"center": [54.8, 97.3625], "radius": 4},
  {"center": [75.2, 98.1625], "radius": 4},
  {"center": [76.8, 92.56249999999999], "radius": 4},
  {"center": [76.6, 86.9625], "radius": 4},
  {"center": [78, 79.5625], "radius": 4},
  {"center": [76.6, 74.3625], "radius": 4},
  {"center": [74.8, 68.7625], "radius": 4},
  {"center": [73.4, 62.1625], "radius": 4},
  {"center": [74, 57.3625], "radius": 4},
  {"center": [72.6, 52.1625], "radius": 4},
  {"center": [71.6, 45.3625], "radius": 4},
  {"center": [70.8, 40.3625], "radius": 4},
  {"center": [70, 35.9625], "radius": 4},
  {"center": [72.6, 30.1625], "radius": 4},
  {"center": [63.4, 27.7625], "radius": 4},
  {"center": [70.2, 23.3625], "radius": 4},
  {"center": [21.2, 31.1625], "radius": 4},
  {"center": [19.6, 37.3625], "radius": 4},
  {"center": [30.6, 21.7625], "radius": 4},
  {"center": [19.6, 22.3625], "radius": 4},
  {"center": [30.4, 65.5625], "radius": 4},
  {"center": [79, 21.9625], "radius": 4},
  {"center": [77.8, 27.7625], "radius": 4},
  {"center": [76, 33.5625], "radius": 4},
  {"center": [81, 30.3625], "radius": 4},
  {"center": [75.6, 37.7625], "radius": 4},
  {"center": [75, 41.9625], "radius": 4},
  {"center": [76.4, 47.9625], "radius": 4}
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
    best_score = frame['gbest']['cost']

    pp.update_path(sol, path_line)
    title.set_text(f"Iteration: {it}, Length: {length:.2f}, Best Score: {best_score:.2f}")
    return path_line, title

# アルゴリズムの選択
obstacles = 'obstacles5'
algorithm = 'ga'  # 'pso', 'abc', 'aco', 'instabae', 'ga' から選択

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

elif algorithm == 'ga':
    # GA parameters
    params = {
        'max_iter': 100,
        'pop_size': 100,
        'crossover_rate': 0.8,
        'mutation_rate': 0.1,
        'mutation_step': 0.1,
        'selection_method': 'tournament',
        'crossover_method': 'uniform',
        'tournament_size': 3,
        'elite_count': 2,
    }
    bestsol, pop = GA_Algorithm(problem, callback=callback, **params)

# なければ保存用ディレクトリを作成
save_dir = f'path-planning/images/{obstacles}'
os.makedirs(save_dir, exist_ok=True)

# アニメーションの作成
ani = animation.FuncAnimation(fig, update, frames=frame_data, init_func=init, blit=False, interval=200, repeat=False)
ani.save(f'path-planning/images/{obstacles}/{algorithm}.gif', writer='pillow')
