import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anime
from sklearn.cluster import KMeans
from osero import PlayOsero

def Evaluate(stoneMap1,stoneMap2):
    n_white,n_black,n_sum = PlayOsero(stoneMap1,stoneMap2)
    return n_white + 8*8 - n_sum

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
    for i in range(8*8):
        vector[i] = max(-30,min(30,vector[i]))

def InitIndivisual():
    return np.random.uniform(-30,30,(8*8))

def EvalIndivisual(vector1,vector2):
    return Evaluate(vector1.reshape((8,8)),vector2.reshape((8,8)))

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
    exe = 0

    def __init__(self, n_iters, n_clusters, n_flies):
        self.n_iters = n_iters
        self.n_clusters = n_clusters
        self.n_flies = n_flies
        self.InitFlies()
        self.EvaluateLikes()

    def InitFlies(self):
        self.vectors = np.array([InitIndivisual() for _ in range(self.n_flies)])
        self.strategies = np.zeros([self.n_flies, 3])
        for i in range(self.n_flies):
            randoms = np.random.uniform(1, 100, 3)
            for j in range(3):
                self.strategies[i][j] = randoms[j]/sum(randoms)

        self.likes = np.zeros(self.n_flies)
        self.best_fly_indices = np.zeros(self.n_clusters)

    def Run(self):
        fig = plt.figure()
        imgs = []
        for i in range(self.n_iters):
            self.EvaluateLikes()
            """
            if not self.centers is None:
                scatter_center = plt.scatter(self.centers.T[0], self.centers.T[1], marker="*",c="red")
                scatter_vector = plt.scatter(self.vectors.T[0], self.vectors.T[1], marker=".",c="blue")
                imgs.append([scatter_center,scatter_vector])
            """
            self.Clustering()
            self.UpdateFlieVector()
        """
        plt.xlim(0,100)
        plt.ylim(0,100)
        plt.grid(True)
        ani = anime.ArtistAnimation(fig, imgs,interval=500)
        ani.save("sample.gif",writer="imagemagick")
        plt.show()
        """

        self.EvaluateLikes()
        best_arg = np.argmax(self.likes)
        return self.likes[best_arg],self.vectors[best_arg]


    def EvaluateLikes(self):
        enemy = np.random.randint(0,self.n_flies,1)
        for i in range(self.n_flies):
            self.exe += 1
            if self.exe%50 == 0:
                print("exe = ",self.exe)
            self.likes[i] = EvalIndivisual(self.vectors[i],self.vectors[enemy])

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
    n_iters = 1000
    n_clusters = 10
    insta = InstaGramFlies(n_iters, n_clusters, n_indivisuals)
    score, result = insta.Run()
    print(score,result)
