import numpy as np
import matplotlib.pyplot as plt


def Pareto(a,mode,shape):
    return (np.random.pareto(a,size=shape)+1)*mode

data = Pareto(10,1,10000)

print(min(data))
print(max(data))

# figureを生成する
fig = plt.figure()

# axをfigureに設定する
ax = fig.add_subplot(1, 1, 1)

# axesにplot
ax.hist(data, bins=50)

# 表示する
plt.show()
# n_cluster = 3
# n_flies = 20
# vectors = np.random.uniform(0,100,n_flies)
# data = {{15.63938386,38.6278939},
#         {73.15622822,30.24212846},
#         {39.96698782,80.50647951}}

# data = np.random.uniform(0,np.pi,200)
# fig = plt.figure()

# # axをfigureに設定する
# ax = fig.add_subplot(1, 1, 1)

# # axesにplot
# ax.hist(data, bins=50)

# # 表示する
# plt.show()

# speeds = np.zeros(100)
# print(speeds)
