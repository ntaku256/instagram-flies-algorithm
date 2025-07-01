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

# アニメーションの作成
ani = animation.FuncAnimation(fig, update, frames=frame_data, init_func=init, blit=False, interval=200, repeat=False)
ani.save(f'path-planning/images/path_planning/{algorithm}.gif', writer='pillow')

# === ここからpop[i]['best']の2Dアニメーション（cost色分け, global/local, FuncAnimation版） ===
all_x_best = [p['best']['position'][0] for frame in frame_data for p in frame['pop']]
all_y_best = [p['best']['position'][1] for frame in frame_data for p in frame['pop']]
xmin_best, xmax_best = min(all_x_best), max(all_x_best)
ymin_best, ymax_best = min(all_y_best), max(all_y_best)

for scale_mode in ['global', 'local']:
    if scale_mode == 'global':
        all_costs = [p['best']['cost'] for frame in frame_data for p in frame['pop']]
        vmin = min(all_costs)
        vmax = max(all_costs)

    fig2d, ax2d = plt.subplots(figsize=(7, 7))

    def init_best2d():
        ax2d.cla()
        ax2d.set_xlabel('x')
        ax2d.set_ylabel('y')
        ax2d.grid(True)
        ax2d.set_xlim(xmin_best, xmax_best)
        ax2d.set_ylim(ymin_best, ymax_best)
        pop = frame_data[0]['pop']
        xs = [p['best']['position'][0] for p in pop]
        ys = [p['best']['position'][1] for p in pop]
        cs = [p['best']['cost'] for p in pop]
        if scale_mode == 'global':
            sc = ax2d.scatter(xs, ys, c=cs, cmap='jet', vmin=vmin, vmax=vmax, s=20)
        else:
            vmin_local = min(cs)
            vmax_local = max(cs)
            sc = ax2d.scatter(xs, ys, c=cs, cmap='jet', vmin=vmin_local, vmax=vmax_local, s=20)
        gbest_pos = frame_data[0]['gbest']['position']
        gbest_sc = ax2d.scatter(gbest_pos[0], gbest_pos[1], c='yellow', edgecolors='black', s=100, marker='*', zorder=10)
        it = frame_data[0]['it']
        length = frame_data[0]['gbest']['details'].get('length', 0)
        ax2d.set_title(f"Iteration: {it}, Length: {length:.2f}")
        return sc, gbest_sc

    def update_best2d(i):
        ax2d.cla()
        ax2d.set_xlabel('x')
        ax2d.set_ylabel('y')
        ax2d.grid(True)
        ax2d.set_xlim(xmin_best, xmax_best)
        ax2d.set_ylim(ymin_best, ymax_best)
        frame = frame_data[i]
        pop = frame['pop']
        xs = [p['best']['position'][0] for p in pop]
        ys = [p['best']['position'][1] for p in pop]
        cs = [p['best']['cost'] for p in pop]
        if scale_mode == 'global':
            sc = ax2d.scatter(xs, ys, c=cs, cmap='jet', vmin=vmin, vmax=vmax, s=20)
        else:
            vmin_local = min(cs)
            vmax_local = max(cs)
            sc = ax2d.scatter(xs, ys, c=cs, cmap='jet', vmin=vmin_local, vmax=vmax_local, s=20)
        gbest_pos = frame['gbest']['position']
        gbest_sc = ax2d.scatter(gbest_pos[0], gbest_pos[1], c='yellow', edgecolors='black', s=100, marker='*', zorder=10)
        it = frame['it']
        length = frame['gbest']['details'].get('length', 0)
        ax2d.set_title(f"Iteration: {it}, Length: {length:.2f}")
        return sc, gbest_sc

    if scale_mode == 'global':
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cbar = fig2d.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='jet'), ax=ax2d)
        cbar.set_label('Cost')

    ani2d = animation.FuncAnimation(fig2d, update_best2d, frames=len(frame_data), init_func=init_best2d, blit=False, interval=300, repeat=False)
    ani2d.save(f'path-planning/images/path_planning_best/{scale_mode}/{algorithm}.gif', writer='pillow')
    plt.close(fig2d)
# === ここまで追加 ===

# === ここからpop[i]['position']の2Dアニメーション（cost色分け, global/local, FuncAnimation版） ===
all_x = [p['position'][0] for frame in frame_data for p in frame['pop']]
all_y = [p['position'][1] for frame in frame_data for p in frame['pop']]
xmin, xmax = min(all_x), max(all_x)
ymin, ymax = min(all_y), max(all_y)

for scale_mode in ['global', 'local']:
    if scale_mode == 'global':
        all_costs = [p['cost'] for frame in frame_data for p in frame['pop']]
        vmin = min(all_costs)
        vmax = max(all_costs)

    fig_pos, ax_pos = plt.subplots(figsize=(7, 7))

    def init_pos():
        ax_pos.cla()
        ax_pos.set_xlabel('x')
        ax_pos.set_ylabel('y')
        ax_pos.grid(True)
        ax_pos.set_xlim(xmin, xmax)
        ax_pos.set_ylim(ymin, ymax)
        pop = frame_data[0]['pop']
        xs = [p['position'][0] for p in pop]
        ys = [p['position'][1] for p in pop]
        cs = [p['cost'] for p in pop]
        if scale_mode == 'global':
            sc = ax_pos.scatter(xs, ys, c=cs, cmap='jet', vmin=vmin, vmax=vmax, s=20)
        else:
            vmin_local = min(cs)
            vmax_local = max(cs)
            sc = ax_pos.scatter(xs, ys, c=cs, cmap='jet', vmin=vmin_local, vmax=vmax_local, s=20)
        gbest_pos = frame_data[0]['gbest']['position']
        gbest_sc = ax_pos.scatter(gbest_pos[0], gbest_pos[1], c='red', edgecolors='black', s=100, marker='*', zorder=10)
        it = frame_data[0]['it']
        length = frame_data[0]['gbest']['details'].get('length', 0)
        ax_pos.set_title(f"Iteration: {it}, Length: {length:.2f}")
        return sc, gbest_sc

    def update_pos(i):
        ax_pos.cla()
        ax_pos.set_xlabel('x')
        ax_pos.set_ylabel('y')
        ax_pos.grid(True)
        ax_pos.set_xlim(xmin, xmax)
        ax_pos.set_ylim(ymin, ymax)
        frame = frame_data[i]
        pop = frame['pop']
        xs = [p['position'][0] for p in pop]
        ys = [p['position'][1] for p in pop]
        cs = [p['cost'] for p in pop]
        if scale_mode == 'global':
            sc = ax_pos.scatter(xs, ys, c=cs, cmap='jet', vmin=vmin, vmax=vmax, s=20)
        else:
            vmin_local = min(cs)
            vmax_local = max(cs)
            sc = ax_pos.scatter(xs, ys, c=cs, cmap='jet', vmin=vmin_local, vmax=vmax_local, s=20)
        gbest_pos = frame['gbest']['position']
        gbest_sc = ax_pos.scatter(gbest_pos[0], gbest_pos[1], c='red', edgecolors='black', s=100, marker='*', zorder=10)
        it = frame['it']
        length = frame['gbest']['details'].get('length', 0)
        ax_pos.set_title(f"Iteration: {it}, Length: {length:.2f}")
        return sc, gbest_sc

    if scale_mode == 'global':
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cbar = fig_pos.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='jet'), ax=ax_pos)
        cbar.set_label('Cost')

    ani_pos = animation.FuncAnimation(fig_pos, update_pos, frames=len(frame_data), init_func=init_pos, blit=False, interval=300, repeat=False)
    ani_pos.save(f'path-planning/images/path-planning_pos/{scale_mode}/{algorithm}.gif', writer='pillow')
    plt.close(fig_pos)
# === ここまで追加 ===

