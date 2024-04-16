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


class PSO:
    n_iter = None
    n_swarm = None
    w = None
    c1 = None
    c2 = None
    vectors = None
    scores = None
    g_best_score = None
    g_best_vector = None
    p_best_scores = None
    p_best_vectors = None
    exe = 0

    def __init__(self,n_iter,n_swarm,w,c1,c2):
        self.n_iter = n_iter
        self.n_swarm = n_swarm
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.InitSwarms()
        self.CalcScores()

    def InitSwarms(self):
        self.vectors = np.array([InitIndivisual() for _ in range(self.n_swarm)])
        self.speeds = np.zeros_like(self.vectors)
        self.p_best_scores = np.zeros(self.n_swarm)
        self.p_best_vectors = np.zeros_like(self.vectors)
        self.g_best_score = 0
        self.g_best_vector= np.zeros_like(self.vectors[0])
        self.scores = np.zeros(self.n_swarm)
    
    def CalcScores(self):
        enemy = np.random.randint(0,self.n_swarm,1)
        for i in range(self.n_swarm):
            self.exe += 1
            if self.exe%50 == 0:
                print("exe = ",self.exe)
            new_score = EvalIndivisual(self.vectors[i],self.vectors[enemy])
            if new_score > self.p_best_scores[i]:
                self.p_best_scores[i] = new_score
                self.p_best_vectors[i] = np.copy(self.vectors[i])
            if new_score > self.g_best_score:
                self.g_best_score = new_score
                self.g_best_vector = np.copy(self.vectors[i])
            self.scores[i] = new_score

    def UpdateVectors(self):
        for i in range(self.n_swarm):
            r1 = np.random.uniform(0,1,self.vectors[0].shape)
            r2 = np.random.uniform(0,1,self.vectors[0].shape)
            self.speeds[i] = self.w*self.speeds[i]+r1*(self.p_best_vectors[i]-self.vectors[i])+r2*(self.g_best_vector-self.vectors[i])
            self.vectors[i] = self.vectors[i] + self.speeds[i]

    def Run(self):
        for i in range(self.n_iter):
            self.CalcScores()
            self.UpdateVectors()
        return self.g_best_score,self.g_best_vector


if __name__ == "__main__":
    n_indivisuals = 15
    n_iters = 1
    c1 = 0.7
    c2 = 0.7
    w = 0.9
    pso = PSO(n_iters,n_indivisuals,w,c1,c2)
    score,result = pso.Run()
    print(score,result)