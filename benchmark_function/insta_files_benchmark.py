import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anime
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

from benchmark_function import BenchmarkFunction as BF

bf = BF()

def Evaluate(x,y):
    return 50 - (x*x + y*y)

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

def EvalIndivisual(x,y):
    return Evaluate(x,y)

class InstaGramFlies:
    n_iters = None
    n_clusters = None
    n_flies = None
    centers = None
    r = None
    # best fly in in each cluster
    best_fly_indices = None
    cluster_like_average = None
    center_dist_average = None
    center_speeds = None

    likes = None
    labels = None
    #np.array([[pioneer rate, faddist rate, master rate],[...],...])
    strategies = None
    vectors = None  # np.array([x1,x2,x3,...]) x1: np.array
    p_best_vectors = None
    p_best_score = None

    def __init__(self, n_iters, n_clusters, n_flies,r):
        self.n_iters = n_iters
        self.n_clusters = n_clusters
        self.n_flies = n_flies
        self.r = r
        self.InitFlies()
        self.EvaluateLikes()

    def InitFlies(self):
        self.vectors = np.array([InitIndivisual(self.r) for _ in range(self.n_flies)])
        self.p_best_vectors = np.zeros_like(self.vectors)
        self.p_best_score = np.zeros(self.n_flies)
        self.strategies = np.zeros([self.n_flies, 3])
        for i in range(self.n_flies):
            randoms = np.random.uniform(1, 100, 3)
            for j in range(3):
                self.strategies[i][j] = randoms[j]/sum(randoms)

        self.likes = np.zeros(self.n_flies)
        self.best_fly_indices = np.zeros(self.n_clusters)

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
        Z = bf.sphere(input_array)
        ZZ = Z.reshape(d)
        imgs = []

        for i in range(self.n_iters):
            self.EvaluateLikes()
            if not self.centers is None:
                title = ax.text(0,0,80,  '%s' % (str(i)), size=20, zorder=1,  color='k') 
                scatter_vector = ax.scatter(self.p_best_vectors.T[0], self.p_best_vectors.T[1],bf.sphere(self.p_best_vectors),c="green", s=10, alpha=1)
                scatter_func = ax.plot_surface(XX, YY, ZZ, rstride = 1, cstride = 1, cmap = plt.cm.coolwarm,alpha=0.55)
                imgs.append([scatter_vector,title])

            self.Clustering()
            self.UpdateFlieVector()

        ani = anime.ArtistAnimation(fig, imgs,interval=500)
        ani.save("benchmark_function/pso.gif",writer="imagemagick")
        plt.show()
        self.EvaluateLikes()
        best_arg = np.argmax(self.likes)
        return self.likes[best_arg],self.vectors[best_arg]

    def EvaluateLikes(self):
        for i in range(self.n_flies):
            self.likes[i] = EvalIndivisual(self.vectors[i][0],self.vectors[i][1])

    def Clustering(self):
        if self.centers is None:
            model = KMeans(n_clusters=self.n_clusters)
            result = model.fit(self.vectors)
            self.centers = result.cluster_centers_
        else:
            model = KMeans(n_init=1,n_clusters=self.n_clusters,init=self.centers)
            result = model.fit(self.vectors)
        self.labels = result.labels_
        self.center_speeds = result.cluster_centers_ - self.centers
        self.centers = result.cluster_centers_
    
        # best flies in each cluster
        best = np.zeros(self.n_clusters)
        self.cluster_like_average = np.zeros(self.n_clusters)
        for i in range(self.n_flies):
            label = self.labels[i]
            if (self.likes[i] > best[label]):
                best[label] = self.likes[i]
                self.best_fly_indices[label] = i
            if (self.likes[i] > self.p_best_score[i]):
                self.p_best_score[i] = self.likes[i]
                self.p_best_vectors[i] = self.vectors[i]

        # like average in each cluster
        for i in range(self.n_clusters):
            self.cluster_like_average[i] = np.mean(
                    [self.likes[j] for j in range(self.n_flies) if self.labels[j] == i])

        # average dist between each cluster
        self.center_dist_average = np.zeros_like(self.vectors[0])
        for i in range(self.n_clusters):
            for j in range(i+1,self.n_clusters):
                self.center_dist_average += (self.centers[i]-self.centers[j])
        self.center_dist_average /= sum(range(1,self.n_clusters))

    def UpdateFlieVector(self):
        for i in range(self.n_flies):
            action = roulett(self.strategies[i])
            # pioneer
            if action == 0:
                self.vectors[i] = self.UpdatePioneer(self.vectors[i])
            # faddist
            if action == 1:
                self.vectors[i] = self.UpdateFaddist(self.vectors[i])
            # master
            if action == 2:
                self.vectors[i] = self.UpdateMaster(self.vectors[i], self.labels[i])
            filtIndivisual(self.vectors[i],self.r)

    def UpdatePioneer(self, vector):
        length = Pareto(1,6,self.center_dist_average.shape)
        rand01 = np.random.choice([-1,1],length.shape)
        return vector + self.center_dist_average*length*rand01

    def UpdateFaddist(self, vector):
        cluster = roulett(self.cluster_like_average)
        return self.UpdateMaster(vector,cluster)

    def UpdateMaster(self, vector, label):
        index_table = [i for i in range(self.n_flies) if(self.labels[i]==label)]
        table = [self.likes[i] for i in index_table]
        target_fly_index = index_table[roulett(table)]
        center = self.centers[label]
        center_vector = (center - vector)*np.random.uniform(0,1)
        target_vector = (self.vectors[target_fly_index] - vector)*np.random.uniform(0,1)
        center_speed_vector = self.center_speeds[label]*np.random.uniform(0,1)
        return vector+center_vector+target_vector+center_speed_vector

if __name__ == "__main__":
    n_indivisuals = 150
    n_iters = 10
    n_clusters = 10
    r = 5
    insta = InstaGramFlies(n_iters, n_clusters, n_indivisuals,r)
    score, result = insta.Run()
    print(score,result)
