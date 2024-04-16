import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anime
from sklearn.cluster import KMeans
from insta_flies import InstaGramFlies
from PSO import PSO
from osero import PlayOsero


if __name__ == "__main__":
    n_indivisuals = 150
    n_iters = 10
    n_clusters = 10
    c1 = 0.7
    c2 = 0.7
    w = 0.9
    pso = PSO(n_iters,n_indivisuals,w,c1,c2)
    instaFlies = InstaGramFlies(n_iters, n_clusters, n_indivisuals)
    stoneMap = []
    flag = True
    # pso,if

    result = []

    for i in range(10):
        print("PSO run")
        _,pso_res = pso.Run()
        print("InstaFiles run")
        _, if_res = instaFlies.Run()
        if flag:
            n_white,n_black,_ = PlayOsero(pso_res.reshape((8,8)),if_res.reshape((8,8)))
            result.append([n_white,n_black])
        else:
            n_white,n_black,_ = PlayOsero(if_res.reshape((8,8)),pso_res.reshape((8,8)))
            result.append([n_black,n_white])
        stoneMap.append(if_res.reshape((8,8)))
        flag = not flag
    print(result)

    insta = 0
    pso = 0
    draw = 0
    for i in range(len(result)):
        if result[i][0]<result[i][1]:
            insta = insta + 1
        elif result[i][0]>result[i][1]:
            pso = pso + 1
        else:
            draw = draw + 1
    print(stoneMap)
    print("insta = ",insta)
    print("pso = ",pso)
    print("draw = ",draw)
