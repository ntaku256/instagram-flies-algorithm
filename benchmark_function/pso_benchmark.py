import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anime
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

from benchmark_function import BenchmarkFunction as BF

bf = BF()

def Evaluate(vector):
    return bf.rosenbrock_star(vector)

def Pareto(mode,a,shape):
    return (np.random.pareto(a,size=shape)+1)*mode

def roulett(table):
    total = np.sum(table)
    rand = np.random.uniform(0.0,total)
    sum = 0
    for i in range(len(table)):
        sum += table[i]
        if(sum > rand):
            return i

def filtIndivisual(vector,r):
    for i in range(1*2):
        vector[i] = max(-r,min(r,vector[i]))

def InitIndivisual(r):
    return np.random.uniform(-r,r,(1*2))

def EvalIndivisual(vector):
    return 3500-Evaluate(vector)


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

    def __init__(self,n_iter,n_swarm,w,c1,c2,r):
        self.n_iter = n_iter
        self.n_swarm = n_swarm
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.r = r
        self.InitSwarms()
        self.CalcScores()

    def InitSwarms(self):
        self.vectors = np.array([InitIndivisual(self.r) for _ in range(self.n_swarm)])
        self.speeds = np.zeros_like(self.vectors)
        self.p_best_scores = np.zeros(self.n_swarm)
        self.p_best_vectors = np.zeros_like(self.vectors)
        self.g_best_score = 0
        self.g_best_vector= np.zeros_like(self.vectors[0])
        self.scores = np.zeros(self.n_swarm)
    
    def CalcScores(self):
        print(EvalIndivisual(self.vectors))
        new_score = EvalIndivisual(self.vectors)
        for i in range(self.n_swarm):
            if new_score[i] > self.p_best_scores[i]:
                self.p_best_scores[i] = new_score[i]
                self.p_best_vectors[i] = np.copy(self.vectors[i])
            if new_score[i] > self.g_best_score:
                self.g_best_score = new_score[i]
                self.g_best_vector = np.copy(self.vectors[i])
            self.scores[i] = new_score[i]

    def UpdateVectors(self):
        for i in range(self.n_swarm):
            r1 = np.random.uniform(0,1,self.vectors[0].shape)
            r2 = np.random.uniform(0,1,self.vectors[0].shape)
            self.speeds[i] = self.w*self.speeds[i]+r1*(self.p_best_vectors[i]-self.vectors[i])+r2*(self.g_best_vector-self.vectors[i])
            self.vectors[i] = self.vectors[i] + self.speeds[i]

    def Run(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f')
        fig.add_axes(ax)
        X = np.linspace(-self.r, self.r, 100)
        Y = np.linspace(-self.r, self.r, 100)
        XX, YY = np.meshgrid(X, Y)
        d = XX.shape
        input_array = np.vstack((XX.flatten(), YY.flatten())).T
        Z = Evaluate(input_array)
        ZZ = Z.reshape(d)
        imgs = []

        for i in range(self.n_iter):
            if i != 0:
                title = ax.text(0,0,5600,  '%s' % (str(i)), size=20, zorder=1,  color='k') 
                scatter_vector = ax.scatter(self.p_best_vectors.T[0], self.p_best_vectors.T[1],Evaluate(self.p_best_vectors),c="green", s=10, alpha=1)
                scatter_func = ax.plot_surface(XX, YY, ZZ, rstride = 1, cstride = 1, cmap = plt.cm.coolwarm,alpha=0.7)
                imgs.append([scatter_vector,scatter_func,title])

            self.CalcScores()
            self.UpdateVectors()
        
        ani = anime.ArtistAnimation(fig, imgs,interval=500)
        ani.save("benchmark_function/pso_rosenbrock_star.gif",writer="imagemagick")
        plt.show()
        return self.g_best_score,self.g_best_vector

if __name__ == "__main__":
    n_indivisuals = 150
    n_iters = 15
    c1 = 0.7
    c2 = 0.7
    w = 0.9
    r = 2
    pso = PSO(n_iters,n_indivisuals,w,c1,c2,r)
    score,result = pso.Run()
    print(score,result)
