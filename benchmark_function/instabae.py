import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.animation as anime
import openpyxl
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist

from benchmark_function import BenchmarkFunction as BF

bf = BF()


def SelectOptimization():
    optimization_problems = {
        "sphere": [-5.12, 5.12],
        "ellipsoid": [-5.12, 5.12],
        "k_tablet": [-5.12, 5.12],
        "rosenbrock_star": [-2.048, 2.048],
        "rosenbrock_chain": [-2.048, 2.048],
        "bohachevsky": [-5.12, 5.12],
        "ackley": [-32.768, 32.768],
        "schaffer": [-100, 100],
        "rastrigin": [-5.12, 5.12],
    }
    print("次の最適化問題から選んでください:")
    for index, problem in enumerate(optimization_problems.keys(), start=1):
        print(f"{index}. {problem}")
    while True:
        try:
            user_input = int(input("選択したい最適化問題の番号を入力してください: "))
            if 1 <= user_input <= len(optimization_problems):
                problem_name = list(optimization_problems.keys())[user_input - 1]
                bounds = optimization_problems[problem_name]
                print(f"選択された最適化問題: {problem_name}\n範囲: {bounds}")
                return problem_name, optimization_problems[problem_name][0],optimization_problems[problem_name][1]
            else:
                print("無効な選択です。1から{}の番号を入力してください。".format(len(optimization_problems)))
        except ValueError:
            print("無効な入力です。番号を整数として入力してください。")


# 評価値の計算
def Evaluate(problem_name, vector):
    match problem_name:
        case "sphere":
            return bf.sphere(vector)
        case "ellipsoid":
            return bf.ellipsoid(vector)
        case "k_tablet":
            return bf.k_tablet(vector)
        case "rosenbrock_star":
            return bf.rosenbrock_star(vector)
        case "rosenbrock_chain":
            return bf.rosenbrock_chain(vector)
        case "bohachevsky":
            return bf.bohachevsky(vector)
        case "ackley":
            return bf.ackley(vector)
        case "schaffer":
            return bf.schaffer(vector)
        case "rastrigin":
            return bf.rastrigin(vector)


# 配列の中から、値の大きさに応じた確率(大きさの比)でインデックスを選択する
def roulett(table):
    rand = np.random.uniform(0.0, np.sum(table))
    sum = 0
    for i in range(len(table)):
        sum += table[i]
        if sum > rand:
            return i


# 配列の中から、値の大きさに応じた確率(逆数の比)でインデックスを選択する
# ただし、0付近で値が極端に大きくなるのでオフセットを使用する
# オフセットの値は、求める解の精度と扱える精度を考慮して決める
def roulettMin(table, offset = 1e-300):
    weights = [1 / (value + offset) for value in table]
    total_weight = sum(weights)
    probabilities = [weight / total_weight for weight in weights]
    return np.random.choice(len(table), p=probabilities)


# 粒子の位置が範囲外に出た場合は、範囲内の最大値にする
def filtIndivisual(vector, r_min,r_max):
    for i in range(len(vector)):
        vector[i] = max(r_min, min(r_max, vector[i]))


# 粒子の位置を初期化
def InitIndivisual(r_min, r_max):
    return np.random.uniform(r_min, r_max, (1 * 2))


class Instabae:
    t_best_score = []

    view = None
    iteration = None
    n_iters = None  # 試行回数
    n_indivisuals = None  # 粒子の数
    n_clusters = None  # クラスタ数

    r_min = None  # 探索範囲
    r_max = None 
    w = None  # 速度の慣性
    c1 = None
    c2 = None
    top_percent = 0.05  # 初期クラスタ数(%)
    max_clusters = 20  # 最大クラスタ数

    min_interest = 0.600  # 関心度の最低値
    max_interest = 1.000  # 関心度の最大値
    min_change_interst = 0.3  # 関心度を減算するときの係数の最低値
    max_change_interst = 1.8  # 関心度を減算するときの係数の最大値
    p_minus_interest = 0.03  # 個人の記録更新がなかった場合の関心度の減りやすさ
    g_minus_interest = 0.02  # クラスタの重心が更新されなかった場合の関心度の減りやすさ

    particles_position = None  # 粒子の位置
    particles_velocity = None  # 粒子の速度
    particles_value = None  # 粒子の評価値

    particles_best_positon = None  # 粒子内の最適解の位置
    particles_best_value = None  # 粒子内の最適解の評価値

    particles_label = None  # 粒子が属するクラスタ
    particles_target_label = None  # 粒子が目指すクラスタ
    particles_target_label_interest = None  # クラスタに対しての関心度
    particles_interset_change = None  # 関心度の減りやすさ

    particles_color = None  # 粒子の色デバッグ用
    colors = None  # カラーマップ
    change_cluster = None

    cluster_center_position = None  # クラスタの重心の位置
    cluster_center_value = None  # クラスタの重心の評価値
    cluster_value_average = None  # クラスタの平均評価値

    problem_name = None  # 最適化問題

    def __init__(self, problem_name, view, n_iters, n_indivisuals, r_min,r_max, w, c1, c2):
        self.view = view
        self.problem_name = problem_name
        self.n_iters = n_iters
        self.n_indivisuals = n_indivisuals
        self.r_min = r_min
        self.r_max = r_max
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.InitIndivisuals()
        self.InitEvaluate()
        self.InitClustering()

    def InitIndivisuals(self):
        self.iteration = 0
        self.max_frame = self.n_iters
        self.n_clusters = int(self.n_indivisuals * self.top_percent)
        self.particles_position = np.array([InitIndivisual(self.r_min, self.r_max) for _ in range(self.n_indivisuals)])
        self.particles_velocity = np.zeros_like(self.particles_position)
        #self.particles_value = np.array([float("inf") for _ in range(self.n_indivisuals)])
        #self.particles_best_positon = np.zeros_like(self.particles_position)
        #self.particles_best_value = np.array([float("inf") for _ in range(self.n_indivisuals)])
        #self.particles_target_label_interest = np.zeros_like(self.particles_value)
        self.particles_interset_change = np.random.uniform(self.min_change_interst, self.max_change_interst, self.n_indivisuals)
        self.particles_color = np.array([(0.0, 0.0, 0.0, 0.0) for _ in range(self.n_indivisuals)])
        #self.cluster_center_position = np.zeros_like(self.particles_position)
        #self.cluster_center_value = np.array([float("inf") for _ in range(self.n_indivisuals)])
        self.cluster_value_average = np.array([float("inf") for _ in range(self.n_clusters)])
        self.change_cluster = []

        # 'tab20'で最大20色の見分けやすいカラーマップを取得
        cmap = plt.cm.get_cmap("tab20", self.n_clusters)
        self.colors = [cmap(i) for i in range(self.n_clusters)]

    def InitEvaluate(self):
        self.particles_value = Evaluate(self.problem_name, self.particles_position)
        self.particles_best_value = self.particles_value
        self.particles_best_positon = self.particles_position

    def InitClustering(self):
        top_indices = np.argsort(self.particles_value)[: self.n_clusters]
        self.cluster_center_position = self.particles_position[top_indices]
        self.cluster_center_value = self.particles_value[top_indices]

        distances = cdist(self.particles_position, self.cluster_center_position, metric="euclidean")
        self.particles_label = np.argmin(distances, axis = 1)

        self.particles_target_label = self.particles_label
        self.particles_target_label_interest = np.random.uniform(self.min_interest, self.max_interest, self.n_indivisuals)
        self.ClusterAverageValue()
        self.SetColor()

    def Evaluate(self):
        self.particles_value = Evaluate(self.problem_name, self.particles_position)

        # p_bestの更新
        for i in range(self.n_indivisuals):
            if self.particles_value[i] < self.particles_best_value[i]:
                self.particles_best_value[i] = self.particles_value[i]
                self.particles_best_positon[i] = self.particles_position[i]
                self.ReduceInterest(i, self.p_minus_interest, 3)
            else:
                self.ReduceInterest(i, self.p_minus_interest, -1)

    def ClusterAverageValue(self):
        for i in range(self.n_clusters):
            self.cluster_value_average[i] = np.mean([self.particles_value[j] for j in range(self.n_indivisuals) if self.particles_label[j] == i])
        self.cluster_value_average = np.nan_to_num(self.cluster_value_average, nan=0.0)

    def SetColor(self):
        for cluster_id in range(self.n_clusters):
            self.particles_color[self.particles_target_label == cluster_id] = self.colors[cluster_id]

    def ReduceInterest(self, i, r_max, coefficient):
        changeInterst = self.particles_interset_change[i] * np.random.uniform(0, r_max) * coefficient
        self.particles_target_label_interest[i] = self.particles_target_label_interest[i] + changeInterst
        self.particles_target_label_interest[i] = max(self.min_interest, min(self.max_interest, self.particles_target_label_interest[i]))

    def UpdateVectors(self):
        for i in range(self.n_indivisuals):
            label = self.particles_target_label[i]
            c1 = np.random.uniform(0, 1, self.particles_position[0].shape) * self.c1
            c2 = np.random.uniform(0, 1, self.particles_position[0].shape) * self.c2
            center_vector = self.cluster_center_position[label] - self.particles_position[i]
            local_vector = self.particles_best_positon[i] - self.particles_position[i]
            self.particles_velocity[i] = self.w * self.particles_velocity[i] + c1 * center_vector + c2 * local_vector
            self.particles_position[i] = self.particles_position[i] + self.particles_velocity[i]
            filtIndivisual(self.particles_position[i], self.r_min, self.r_max)

    def Clustering(self):
        # クラスタの重心点を更新
        for i in range(self.n_clusters):
            isUpdate = False
            particles_index = np.flatnonzero(self.particles_label == i)
            for index in particles_index:
                if self.particles_value[index] < self.cluster_center_value[i]:
                    self.cluster_center_value[i] = self.particles_value[index]
                    self.cluster_center_position[i] = self.particles_position[index]
                    isUpdate = True
            if isUpdate:
                for index in particles_index:
                    self.ReduceInterest(index, self.p_minus_interest, 1)
            else:
                for index in particles_index:
                    self.ReduceInterest(index, self.p_minus_interest, -1)

        # クラスタリング
        distances = cdist(self.particles_position, self.cluster_center_position, metric="euclidean")
        self.particles_label = np.argmin(distances, axis=1)    
        self.ClusterAverageValue()

        # 目標クラスタの変更
        count = 0
        if self.n_clusters > 1:
            for i in range(self.n_indivisuals):
                if np.random.uniform(0, 1) >= self.particles_target_label_interest[i]:
                    label = self.particles_target_label[i]
                    j = roulettMin(np.delete(self.cluster_value_average, label))
                    self.particles_target_label[i] = j + (j >= label)
                    self.particles_target_label_interest[i] = np.random.uniform(self.min_interest, self.max_interest)
                    count = count + 1
        self.change_cluster.append(count)
        self.SetColor()


        self.t_best_score.append(min(self.particles_best_value))

        # print(f"count = {count}")
        # print(f"{self.particles_target_label_interest[80]:.3g}%,label = {self.particles_target_label[80]}")

    def Run(self):
        fig = plt.figure(figsize=(6, 6), dpi=200)
        if self.view == "2D":
            ax = fig.add_subplot(111)
        elif self.view == "3D":
            ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if self.view == "3D":
            ax.set_zlabel("f")
            ax.view_init(elev=45, azim=-60)
        fig.add_axes(ax)
        X = np.linspace(self.r_min, self.r_max, 100)
        Y = np.linspace(self.r_min, self.r_max, 100)
        XX, YY = np.meshgrid(X, Y)
        d = XX.shape
        input_array = np.vstack((XX.flatten(), YY.flatten())).T
        Z = Evaluate(self.problem_name, input_array)
        ZZ = Z.reshape(d)
        if self.view == "2D":
            ax.contourf(XX, YY, ZZ, levels=1000, cmap=cm.coolwarm, alpha=0.2)
            scatter_particles = ax.scatter(self.particles_position.T[0], self.particles_position.T[1], s=4, c=self.particles_color,  alpha=0.7)
            scatter_center = ax.scatter(self.cluster_center_position.T[0], self.cluster_center_position.T[1], s=6,  c="black",  alpha=0.4)
        elif self.view == "3D":
            ax.plot_surface(XX, YY, ZZ, rstride=1, cstride=1, cmap = plt.cm.coolwarm, alpha=0.6)
            scatter_particles = ax.scatter(self.particles_position.T[0], self.particles_position.T[1], self.particles_value, c=self.particles_color, s=2, alpha=1)
            scatter_center = ax.scatter(self.cluster_center_position.T[0], self.cluster_center_position.T[1], self.cluster_center_value, c="black", s=4, alpha=0.6)

        def init():
            pass

        def update(i):
            if self.iteration >= self.n_iters:
                ax.set_title(f"Iteration: {self.iteration}", fontsize=12)
                if self.view == "2D":
                    scatter_particles.set_offsets(np.c_[self.particles_position.T[0], self.particles_position.T[1]])
                    scatter_center.set_offsets(np.c_[self.cluster_center_position.T[0], self.cluster_center_position.T[1]])
                elif self.view == "3D":
                    scatter_particles._offsets3d = (self.particles_position.T[0], self.particles_position.T[1], self.particles_value)
                    scatter_center._offsets3d = (self.cluster_center_position.T[0], self.cluster_center_position.T[1], self.cluster_center_value)
                return 

            while not(self.iteration in frames) and self.iteration < self.n_iters - 1:
                self.UpdateVectors()
                self.Evaluate()
                self.Clustering()
                self.iteration += 1

            ax.set_title(f"Iteration: {self.iteration}", fontsize=12)
            if self.view == "2D":
                scatter_particles.set_offsets(np.c_[self.particles_position.T[0], self.particles_position.T[1]])
                scatter_center.set_offsets(np.c_[self.cluster_center_position.T[0], self.cluster_center_position.T[1]])
            elif self.view == "3D":
                scatter_particles._offsets3d = (self.particles_position.T[0], self.particles_position.T[1], self.particles_value)
                scatter_center._offsets3d = (self.cluster_center_position.T[0], self.cluster_center_position.T[1], Evaluate(self.problem_name, self.cluster_center_position))
            self.UpdateVectors()
            self.Evaluate()
            self.Clustering()
            self.iteration += 1

        frames = []
        frame = 0
        while(frame <= self.n_iters):
            if frame <= 10 or (frame) % 50 == 0:
                frames.append(frame)
            frame += 1
    
        ani = anime.FuncAnimation(fig, update, frames=len(frames),init_func=init, interval=300 , repeat=False)
        ani.save(f"benchmark_function/GIF/instabae/{self.view}/{self.problem_name}.gif", writer="imagemagick", dpi=200)

        best_arg = np.argmin(self.particles_best_value)
        print(f"best = {min(self.particles_best_value)}, position = {self.particles_best_positon[best_arg]}, change_ave = {np.mean(self.change_cluster)}")

        return self.t_best_score


if __name__ == "__main__":
    n_iters = 300
    n_indivisuals = 150

    w = 0.4
    c1 = 0.7
    c2 = 0.4

    view = "3D"

    problem_name, r_min,r_max = SelectOptimization()

    instabae = Instabae(problem_name, view, n_iters, n_indivisuals, r_min,r_max, w, c1, c2)
    t_best_score = instabae.Run()

    # write_wb = openpyxl.load_workbook("results/Book_write_1.xlsx")
    # write_ws = write_wb["Sheet1"]

    # # シートを初期化
    # for row in write_ws:
    #     for cell in row:
    #         cell.value = None

    # for j in range(len(t_best_score)):
    #     c = write_ws.cell(j+1 , 1)
    #     c.value = t_best_score[j]
            
    # write_wb.save("results/Book_write_1.xlsx")