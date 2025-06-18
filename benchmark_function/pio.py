import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anime
import openpyxl
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

from benchmark_function import BenchmarkFunction as BF

bf = BF()

def Evaluate(vector):
    return bf.sphere(vector)
# パレート分布
def Pareto(mode,a,shape): 
    return (np.random.pareto(a,size=shape)+1)*mode

def UpdatePioneer(vector,center_dist_average):
        length = Pareto(1,6,center_dist_average.shape)
        rand01 = np.random.choice([-1,1],length.shape)
        return vector + center_dist_average*length*rand01

def filtIndivisual(vector,r):
    for i in range(1*2):
        vector[i] = max(-r,min(r,vector[i]))

def Clustering(vectors):
        centers = None
        n_clusters = 3
        if centers is None:
            model = KMeans(n_clusters=n_clusters) #クラスタリング(グループ分け)
            result = model.fit(vectors) #平均を求める
            centers = result.cluster_centers_
        else:
            model = KMeans(n_init=1,n_clusters=n_clusters,init=centers)
            result = model.fit(vectors)
        labels = result.labels_
        centers = result.cluster_centers_

        # average dist between each cluster
        center_dist_average = np.zeros_like(vectors[0])
        for i in range(n_clusters):
            for j in range(i+1,n_clusters):
                center_dist_average += (centers[i]-centers[j])
        center_dist_average /= sum(range(1,n_clusters))

        return labels,center_dist_average,centers

if __name__ == "__main__":
        ffig, ax = plt.subplots()

        X = np.linspace(-5.12, 5.12, 100)
        Y = np.linspace(-5.12, 5.12, 100)
        XX, YY = np.meshgrid(X, Y)
        d = XX.shape
        input_array = np.vstack((XX.flatten(), YY.flatten())).T
        Z = Evaluate(input_array)
        ZZ = Z.reshape(d)

        vector = []

        count = 72

        x = 2
        y = 0.2

        for i in range(count):
            vector.append([x*np.sin(np.pi*i*2/count),y*np.cos(np.pi*i*2/count)])

        vectors = np.array(vector)

        labels,center_dist_average,centers = Clustering(vectors)

        pios = []
        for i in range(len(vectors)):
            pio = UpdatePioneer(vectors[i],center_dist_average)
            filtIndivisual(pio,5.12)
            pio = np.atleast_2d(pio)
            pios.append(pio[0])



        piopio = np.array(pios)
        # title = ax.text(0,0  '%s' % (str(i+1)), size=20, zorder=1,  color='k') 
        ax.contourf(XX, YY, ZZ, cmap=plt.cm.coolwarm, alpha=0.25)  # 等高線プロット
        ax.scatter(vectors[:, 0], vectors[:, 1], c="blue", s=20, alpha=0.5)  # 元のベクトルをプロット
        ax.scatter(piopio[:, 0], piopio[:, 1], c="green", s=20, alpha=0.5)  # 更新されたベクトルをプロット
        ax.scatter(centers[:, 0], centers[:, 1], c="black", s=100, alpha=0.7)  # クラスタの中心をプロット

        ax.set_aspect('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()