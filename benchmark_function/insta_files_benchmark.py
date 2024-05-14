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

def filtIndivisual(vector):
    for i in range(1*2):
        vector[i] = max(-5,min(5,vector[i]))

def InitIndivisual():
    return np.random.uniform(-5,5,(1*2))

def EvalIndivisual(x,y):
    return Evaluate(x,y)

class InstaGramFlies:
    n_iters = None
    n_clusters = None
    n_flies = None
 
    centers = None
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

    def __init__(self, n_iters, n_clusters, n_flies):
        self.n_iters = n_iters
        self.n_clusters = n_clusters
        self.n_flies = n_flies
        self.InitFlies()
        self.EvaluateLikes()

    def InitFlies(self):
        self.vectors = np.array([InitIndivisual() for _ in range(self.n_flies)])
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
        # fig, ax = plt.subplots()
        # imgs = []
        for i in range(self.n_iters):
            self.EvaluateLikes()
            """"""
            # if not self.centers is None:
            #     title = plt.text(0.5,1.01,i, ha="center",va="bottom",transform=ax.transAxes, fontsize="large")
            #     # scatter_center = ax.scatter(self.centers.T[0], self.centers.T[1], marker="*",c="red")
            #     # scatter_vector = ax.scatter(self.vectors.T[0], self.vectors.T[1], marker=".",c="blue")
            #     # imgs.append([scatter_center,scatter_vector,title])
            #     scatter_vector = ax.scatter(self.p_best_vectors.T[0], self.p_best_vectors.T[1], marker=".",c="blue")
            #     imgs.append([scatter_vector,title])

            if i == 0 or i == 1 or i == 2 or i == 3 or i == 5 or i%50 == 0:
                self.Visualization(self.p_best_vectors)
            """"""
            self.Clustering()
            self.UpdateFlieVector()
        """"""
        # plt.xlim(-5,5)
        # plt.ylim(-5,5)
        # plt.grid(True)
        # ani = anime.ArtistAnimation(fig, imgs,interval=500)
        # ani.save("benchmark_function/insta_files.gif",writer="imagemagick")
        # plt.show()
        """"""

        self.EvaluateLikes()
        best_arg = np.argmax(self.likes)
        return self.likes[best_arg],self.vectors[best_arg]

    def Visualization(self,vectors):
        fig = plt.figure()
        ax = Axes3D(fig)
        fig.add_axes(ax)

        a = 5.12
        X = np.linspace(-a, a, 100)
        Y = np.linspace(-a, a, 100)
        XX, YY = np.meshgrid(X, Y)
        d = XX.shape
        input_array = np.vstack((XX.flatten(), YY.flatten())).T

        Z = bf.sphere(input_array)
        ZZ = Z.reshape(d)

        ax.scatter(vectors.T[0], vectors.T[1],bf.sphere(vectors),c="green", s=10, alpha=1)
        ax.plot_surface(XX, YY, ZZ, rstride = 1, cstride = 1, cmap = plt.cm.coolwarm,alpha=0.6)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('f')
        plt.show()



    def EvaluateLikes(self):
        enemy = np.random.randint(0,self.n_flies,1)
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
            filtIndivisual(self.vectors[i])

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
    n_iters = 100
    n_clusters = 10
    insta = InstaGramFlies(n_iters, n_clusters, n_indivisuals)
    score, result = insta.Run()
    print(score,result)
