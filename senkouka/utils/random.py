import numpy as np

class Random:

    def roulett(table):
        total = np.sum(table)
        rand = np.random.uniform(0.0, total)
        sum = 0
        for i, value in enumerate(table):
            sum += value
            if sum > rand:
                return i
                
        