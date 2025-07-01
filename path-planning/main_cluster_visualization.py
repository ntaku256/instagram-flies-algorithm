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

# アルゴリズムの選択（クラスタ機能があるアルゴリズムのみ）
algorithm = 'instabae'  # 'instabae_v2' または 'instabae_v2'

if algorithm == 'instabae':
    # Instabae parameters
    params = {
        'max_iter': 100,
        'n_clusters': 10,
        'pop_size': 100,
        'r': 5,
    }
    bestsol, pop = IA_Algorithm(problem, callback=callback, **params)
    num_clusters = params['n_clusters']

elif algorithm == 'instabae_v2':
    params = {
        'max_iter': 100,
        'pop_size': 100,
    }
    bestsol, pop = IA_V2_Algorithm(problem, callback=callback, **params)
    # instabae_v2の場合、クラスタ数を動的に決定するため、実際のクラスタ数を取得
    all_cluster_labels = [p['cluster_label'] for frame in frame_data for p in frame['pop']]
    num_clusters = len(set(all_cluster_labels))

# アニメーションの作成
ani = animation.FuncAnimation(fig, update, frames=frame_data, init_func=init, blit=False, interval=200, repeat=False)
ani.save(f'path-planning/images/path_planning/{algorithm}_cluster.gif', writer='pillow')


print("次元数", problem['num_var'])
print("pop[i]['position']の次元数", len(frame_data[0]['pop'][0]['position']))

# === ここからpop[i]['position']の2Dアニメーション（cluster_label色分け版） ===
all_x = [p['position'][0] for frame in frame_data for p in frame['pop']]
all_y = [p['position'][1] for frame in frame_data for p in frame['pop']]
xmin, xmax = min(all_x), max(all_x)
ymin, ymax = min(all_y), max(all_y)

# クラスタ色分けアニメーション
fig_cluster, ax_cluster = plt.subplots(figsize=(8, 8))

# 離散的なカラーマップを使用（クラスタ用）
colors = plt.cm.tab10(np.linspace(0, 1, num_clusters))

def init_cluster():
    ax_cluster.cla()
    ax_cluster.set_xlabel('x')
    ax_cluster.set_ylabel('y')
    ax_cluster.grid(True)
    ax_cluster.set_xlim(xmin, xmax)
    ax_cluster.set_ylim(ymin, ymax)
    
    pop = frame_data[0]['pop']
    xs = [p['position'][0] for p in pop]
    ys = [p['position'][1] for p in pop]
    cluster_labels = [p['cluster_label'] for p in pop]
    
    # クラスタごとに色分けしてプロット
    scatter_plots = []
    for i in range(num_clusters):
        cluster_x = [x for x, label in zip(xs, cluster_labels) if label == i]
        cluster_y = [y for y, label in zip(ys, cluster_labels) if label == i]
        if cluster_x:  # クラスタに属するパーティクルがある場合のみプロット
            sc = ax_cluster.scatter(cluster_x, cluster_y, c=[colors[i]], s=20, 
                                  label=f'Cluster {i}', alpha=0.7)
            scatter_plots.append(sc)
    
    # クラスタ重心をプロット
    centers_plots = []
    if 'centers' in frame_data[0] and frame_data[0]['centers'] is not None:
        centers = frame_data[0]['centers']
        for i in range(num_clusters):
            if i < len(centers):
                center_sc = ax_cluster.scatter(centers[i][0], centers[i][1], 
                                             c='white', edgecolors=colors[i], 
                                             s=100, marker='o', linewidths=3,
                                             zorder=9, label=f'Center {i}')
                centers_plots.append(center_sc)
    
    # gbestをプロット
    gbest_pos = frame_data[0]['gbest']['position']
    gbest_sc = ax_cluster.scatter(gbest_pos[0], gbest_pos[1], c='red', 
                                edgecolors='black', s=150, marker='*', 
                                zorder=10, label='Global Best')
    
    it = frame_data[0]['it']
    length = frame_data[0]['gbest']['details'].get('length', 0)
    ax_cluster.set_title(f"Cluster Visualization - Iteration: {it}, Length: {length:.2f}")
    ax_cluster.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    return scatter_plots + centers_plots + [gbest_sc]

def update_cluster(i):
    ax_cluster.cla()
    ax_cluster.set_xlabel('x')
    ax_cluster.set_ylabel('y')
    ax_cluster.grid(True)
    ax_cluster.set_xlim(xmin, xmax)
    ax_cluster.set_ylim(ymin, ymax)
    
    frame = frame_data[i]
    pop = frame['pop']
    xs = [p['position'][0] for p in pop]
    ys = [p['position'][1] for p in pop]
    cluster_labels = [p['cluster_label'] for p in pop]
    
    # クラスタごとに色分けしてプロット
    scatter_plots = []
    for j in range(num_clusters):
        cluster_x = [x for x, label in zip(xs, cluster_labels) if label == j]
        cluster_y = [y for y, label in zip(ys, cluster_labels) if label == j]
        if cluster_x:  # クラスタに属するパーティクルがある場合のみプロット
            sc = ax_cluster.scatter(cluster_x, cluster_y, c=[colors[j]], s=20, 
                                  label=f'Cluster {j}', alpha=0.7)
            scatter_plots.append(sc)
    
    # クラスタ重心をプロット
    centers_plots = []
    if 'centers' in frame and frame['centers'] is not None:
        centers = frame['centers']
        for j in range(num_clusters):
            if j < len(centers):
                center_sc = ax_cluster.scatter(centers[j][0], centers[j][1], 
                                             c='white', edgecolors=colors[j], 
                                             s=100, marker='o', linewidths=3,
                                             zorder=9, label=f'Center {j}')
                centers_plots.append(center_sc)
    
    # gbestをプロット
    gbest_pos = frame['gbest']['position']
    gbest_sc = ax_cluster.scatter(gbest_pos[0], gbest_pos[1], c='red', 
                                edgecolors='black', s=150, marker='*', 
                                zorder=10, label='Global Best')
    
    it = frame['it']
    length = frame['gbest']['details'].get('length', 0)
    ax_cluster.set_title(f"Cluster Visualization - Iteration: {it}, Length: {length:.2f}")
    ax_cluster.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    return scatter_plots + centers_plots + [gbest_sc]

# クラスタ色分けアニメーションの作成と保存
ani_cluster = animation.FuncAnimation(fig_cluster, update_cluster, frames=len(frame_data), 
                                    init_func=init_cluster, blit=False, interval=300, repeat=False)
ani_cluster.save(f'path-planning/images/path-planning_pos/cluster/{algorithm}.gif', writer='pillow')

plt.tight_layout()
plt.close(fig_cluster)

print(f"クラスタ可視化アニメーションが保存されました: path-planning/images/path-planning_pos/cluster/{algorithm}.gif")
print(f"使用されたクラスタ数: {num_clusters}") 
