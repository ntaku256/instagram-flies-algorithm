import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anime
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import openpyxl
import mpmath
from benchmark_function import BenchmarkFunction as BF

bf = BF()

def Evaluate(vector):
    # Ensure the vector is reshaped to a 2D array before passing it to the rosenbrock_star function
    # vector = np.reshape(vector, (vector.shape[0], -1))
    return bf.rastrigin(vector)

def filtIndivisual(vector, r):
    for i in range(1 * 2):
        vector[i] = max(-r, min(r, vector[i]))

def InitIndivisual(r):
    return np.random.uniform(-r, r, (1 * 2))

def EvalIndivisual(vector):
    return Evaluate(vector)

class PSO:
    n_iter = None
    n_swarm = None
    w = None
    c1 = None
    c2 = None
    r = None
    vectors = None
    scores = None
    g_best_score = None
    g_best_vector = None
    p_best_scores = None
    p_best_vectors = None
    j = None
    t_g_score = []

    def __init__(self, n_iter, n_swarm, w, c1, c2, r, j):
        self.n_iter = n_iter
        self.n_swarm = n_swarm
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.r = r
        self.j = j
        self.t_g_score = []
        self.InitSwarms()
        self.CalcScores()

    def InitSwarms(self):
        self.vectors = [InitIndivisual(self.r) for _ in range(self.n_swarm)]
        self.speeds = [mpmath.mpf(0) for _ in range(self.n_swarm)]
        self.p_best_scores = [mpmath.mpf('inf') for _ in range(self.n_swarm)]
        self.p_best_vectors = [mpmath.mpf(0) for _ in range(self.n_swarm)]
        self.g_best_score = mpmath.mpf('inf')
        self.g_best_vector = [mpmath.mpf(0) for _ in range(2)]
        self.scores = [mpmath.mpf('inf') for _ in range(self.n_swarm)]
    
    def CalcScores(self):
        new_score = EvalIndivisual(self.vectors)
        for i in range(self.n_swarm):
            if new_score[i] < self.p_best_scores[i]:
                self.p_best_scores[i] = new_score[i]
                self.p_best_vectors[i] = self.vectors[i]
            if new_score[i] < self.g_best_score:
                self.g_best_score = new_score[i]
                self.g_best_vector = self.vectors[i]
            self.scores[i] = new_score[i]

    def UpdateVectors(self):
        for i in range(self.n_swarm):
            r1 = np.random.uniform(0, 1, self.vectors[0].shape)
            r2 = np.random.uniform(0, 1, self.vectors[0].shape)
            self.speeds[i] = self.w * self.speeds[i] + self.c1 * r1 * (self.p_best_vectors[i] - self.vectors[i]) + self.c2 * r2 * (self.g_best_vector - self.vectors[i])
            self.vectors[i] = self.vectors[i] + self.speeds[i]
            filtIndivisual(self.vectors[i], self.r)

    def Run(self):
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('f')
        # fig.add_axes(ax)
        # X = np.linspace(-self.r, self.r, 101)
        # Y = np.linspace(-self.r, self.r, 101)
        # XX, YY = np.meshgrid(X, Y)
        # d = XX.shape
        # input_array = np.vstack((XX.flatten(), YY.flatten())).T
        # Z = Evaluate(input_array)
        # ZZ = Z.reshape(d)
        # imgs = []
        # r= 2.048
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('f')
        # ax.view_init(elev=70, azim=0)
        # fig.add_axes(ax)
        # X = np.linspace(-r, r, 100)
        # Y = np.linspace(-r, r, 100)
        # XX, YY = np.meshgrid(X, Y)
        # d = XX.shape
        # input_array = np.vstack((XX.flatten(), YY.flatten())).T
        # Z = Evaluate(input_array)
        # Z_list = list(map(mpmath.mpf, Z))
        # d = XX.shape
        # ZZ = np.array(Z_list).reshape(d)
        # # ZZ = Z.reshape(d)
        # imgs = []

        for i in range(self.n_iter):
            self.CalcScores()
            self.UpdateVectors()

            # vector = np.array(self.vectors)
            # # if i < 2 or i == 4 or (i + 1) % 50 == 0 or i == self.n_iter - 1:
            # if ((i - 1) < 2 or (i - 1) == 4 or (i)%50  == 0 ):
            #     title = ax.text(0, 0, 1.61 * max(Z), '%s' % (str(i + 1)), size=20, zorder=1, color='k') 
            #     scatter_vector = ax.scatter(vector.T[0], vector.T[1], Evaluate(vector), c="blue", s=3, alpha=1)
            #     scatter_func = ax.plot_surface(XX, YY, ZZ, rstride = 1, cstride = 1, cmap = plt.cm.coolwarm,alpha=0.45)
            #     imgs.append([scatter_vector, scatter_func, title])

            self.t_g_score.append(self.g_best_score)

            self.w = self.w - ((self.w - 0.4) / self.n_iter)

        # ani = anime.ArtistAnimation(fig, imgs, interval=600)
        # ani.save("benchmark_function/GIF/pso/pso.gif", writer="imagemagick")
        # plt.show()

        return self.t_g_score, self.g_best_score, self.g_best_vector

if __name__ == "__main__":
    n_indivisuals = 150
    n_iters = 1000
    c1 = 0.7
    c2 = 0.7
    w = 0.9
    r = 5.12
    best = []
    pso = PSO(n_iters, n_indivisuals, w, c1, c2, r, 0)
    for i in range(1):
        pso = PSO(n_iters, n_indivisuals, w, c1, c2, r, i)
        g_score, score, result = pso.Run()
        best.append(g_score)
        print(score, result)

    # The part where the Excel file was written to has been removed as requested

    write_wb = openpyxl.load_workbook("results/Book_write.xlsx")
    write_ws = write_wb["Sheet1"]

    # シートを初期化
    for row in write_ws:
        for cell in row:
            cell.value = None

    for i in range(len(best)):
        for j in range(len(best[i])):
            c = write_ws.cell(j+1 , i+1)
            c.value = str(best[i][j])
            
    write_wb.save("results/Book_write.xlsx")