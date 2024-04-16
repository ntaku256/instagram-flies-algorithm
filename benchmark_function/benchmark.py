import math

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# User-defined function
from benchmark_function import BenchmarkFunction as BF

bf = BF()

if __name__ == "__main__":
    a = 5.12
    X = np.linspace(-a, a, 101)
    Y = np.linspace(-a, a, 101)
    XX, YY = np.meshgrid(X, Y)
    d = XX.shape

    input_array = np.vstack((XX.flatten(), YY.flatten())).T

    fig = plt.figure()
    ax = Axes3D(fig)
    fig.add_axes(ax)

    Z = bf.sphere(input_array)
    ZZ = Z.reshape(d)

    ax.plot_surface(XX, YY, ZZ, rstride = 1, cstride = 1, cmap = plt.cm.coolwarm)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f')
    plt.show()

    m = 50
    plt.plot(XX[m],ZZ[m])
    plt.xlabel('x1')
    plt.ylabel('f')
    plt.title('x2 = 0')
    plt.show()

    plt.plot(YY.T[m],ZZ.T[m])
    plt.xlabel('x2')
    plt.ylabel('f')
    plt.title('x1 = 0')
    plt.show()