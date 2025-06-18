
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# 目的関数を定義する：ここでは引数 x の各要素の二乗の和を計算します。
def objective_function(x):
    # return np.sum(x**2, axis=-1)

    # A = 10
    # return A * x.shape[-1] + np.sum(x**2 - A * np.cos(2 * np.pi * x), axis=-1)

    # center = np.zeros_like(x)
    # return np.sum(100 * ((x - center[:-1]) ** 2 - (x[1:] - center[:-1] ** 2) ** 2) ** 2 + (x - 1) ** 2, axis=-1)

    result = 0
    for i in range(x.shape[-1] - 1):
        result += 100 * (x[..., i + 1] - x[..., i]**2)**2 + (x[..., i] - 1)**2
    return result


# 人工蜂群アルゴリズム (ABC) の実装
def abc(obj_func, dim, pop_size, max_iter, lb, ub, limit=100, grid_size=50):
    # 蜂の位置と評価値の初期設定
    bee_pos = np.random.uniform(lb, ub, (pop_size, dim))
    bee_scores = np.array([obj_func(bee) for bee in bee_pos])
    # 群れの中で最も優れた位置とその評価値
    best_pos = bee_pos[np.argmin(bee_scores)]
    best_score = np.min(bee_scores)

    # 各蜂の試行回数カウンター
    trial_counter = np.zeros(pop_size)

    # 蜜濃度を記録するグリッドの準備
    grid = np.zeros((grid_size, grid_size))
    grid_x = np.linspace(lb, ub, grid_size)
    grid_y = np.linspace(lb, ub, grid_size)

    # アニメーションで各イテレーションを表示する関数
    def animate(i):
        nonlocal bee_pos, bee_scores, best_pos, best_score, trial_counter, grid

        # 探索蜂のフェーズ
        for j in range(pop_size):
            k = np.random.randint(pop_size)
            while k == j:
                k = np.random.randint(pop_size)
            d = np.random.randint(dim)
            new_pos = bee_pos[j].copy()
            new_pos[d] = bee_pos[j][d] + np.random.uniform(-1, 1) * (bee_pos[j][d] - bee_pos[k][d])
            new_pos = np.clip(new_pos, lb, ub)
            new_score = obj_func(new_pos)
            if new_score < bee_scores[j]:
                bee_pos[j] = new_pos
                bee_scores[j] = new_score
                trial_counter[j] = 0
            else:
                trial_counter[j] += 1

        # 従者蜂のフェーズ
        fitness = np.exp(-bee_scores / np.max(bee_scores))
        probabilities = fitness / np.sum(fitness)
        for j in range(pop_size):
            if np.random.rand() < probabilities[j]:
                k = np.random.randint(pop_size)
                while k == j:
                    k = np.random.randint(pop_size)
                d = np.random.randint(dim)
                new_pos = bee_pos[j].copy()
                new_pos[d] = bee_pos[j][d] + np.random.uniform(-1, 1) * (bee_pos[j][d] - bee_pos[k][d])
                new_pos = np.clip(new_pos, lb, ub)
                new_score = obj_func(new_pos)
                if new_score < bee_scores[j]:
                    bee_pos[j] = new_pos
                    bee_scores[j] = new_score
                    trial_counter[j] = 0
                else:
                    trial_counter[j] += 1

        # 偵察蜂のフェーズ
        for j in range(pop_size):
            if trial_counter[j] >= limit:
                bee_pos[j] = np.random.uniform(lb, ub, dim)
                bee_scores[j] = obj_func(bee_pos[j])
                trial_counter[j] = 0

        # 全体の最良解の更新
        if np.min(bee_scores) < best_score:
            best_score = np.min(bee_scores)
            best_pos = bee_pos[np.argmin(bee_scores)]

        # 蜜濃度を更新
        for pos in bee_pos:
            x_idx = np.digitize(pos[0], grid_x) - 1
            y_idx = np.digitize(pos[1], grid_y) - 1
            grid[y_idx, x_idx] += 1

        # 現在の群れの状況をプロット
        plt.clf()
        plt.imshow(grid.T, extent=(lb, ub, lb, ub), origin="lower", cmap="YlOrBr", alpha=0.8)
        plt.colorbar(label="Honey Density")
        plt.scatter(bee_pos[:, 0], bee_pos[:, 1], c="b", s=4, label="Bees")
        plt.scatter(best_pos[0], best_pos[1], c="r", s=4, label="Best Solution")

        x = np.linspace(lb, ub, 100)
        y = np.linspace(lb, ub, 100)
        X, Y = np.meshgrid(x, y)
        Z = objective_function(np.stack((X, Y), axis=-1))
        plt.contour(X, Y, Z, 20, cmap="coolwarm", alpha=0.2)

        plt.xlim(lb, ub)
        plt.ylim(lb, ub)
        plt.legend()
        plt.title(f"Iteration {i+1}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()

    fig = plt.figure()
    anim = FuncAnimation(fig, animate, frames=max_iter, interval=100)
    anim.save("benchmark_function/GIF/pso/abc_process_with_density.gif", writer="pillow")
    # plt.show()

    return best_pos, best_score


# パラメータ設定とアルゴリズムの実行
dim = 2
pop_size = 150
max_iter = 100
lb = -2.048
ub = 2.048

# lb = -5.12
# ub = 5.12

best_pos, best_score = abc(objective_function, dim, pop_size, max_iter, lb, ub)
print(f"Best position: {best_pos}, Best score: {best_score}")

