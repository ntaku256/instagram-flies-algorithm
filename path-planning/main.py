import path_planning as pp
import matplotlib.pyplot as plt
import matplotlib.animation as anime
from pso import PSO
from abc import ABC as ABCAlgorithm  # この行を削除# filepath: c:\Users\moriken\Documents\instabae-algrithm\instagram-flies-algorithm\path-planning\main.py
from aco import ACO

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

# アニメーション用のフレームを保存するリスト
frames = []

# Callback function
path_line = None
def callback(data):
    global path_line, frames
    it = data['it']
    sol = data['gbest']['details']['sol']
    
    if it == 1:
        fig = plt.figure(figsize=[7, 7])
        pp.plot_environment(env)
        path_line = pp.plot_path(sol, color='b')
        plt.grid(True)
    else:
        pp.update_path(sol, path_line)

    length = data['gbest']['details']['length']
    title = plt.title(f"Iteration: {it}, Length: {length:.2f}")
    
    # フレームの保存
    frames.append([path_line, title])

# アルゴリズムの選択
algorithm = 'abc'  # 'pso', 'abc', 'aco' から選択

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
    bestsol, pop = ABCAlgorithm(problem, callback=callback, **params)

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
    bestsol, pop = ABCAlgorithm(problem, callback=callback, **params)

# アニメーションの作成
ani = anime.ArtistAnimation(plt.gcf(), frames, interval=200, blit=True, repeat=False)
ani.save(f'path-planning\images\path_planning_{algorithm}.gif', writer='pillow')
plt.show()
