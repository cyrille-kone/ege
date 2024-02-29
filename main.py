import numpy as np
from lib.bandits import Bernoulli
def load(ds="C")->np.ndarray:
    if ds=='C':
        return np.load("data/cov_boost.npy")
    elif ds=="S":
        return np.load("data/sort_256.npy")
    elif ds=="I3":
        # group of sub-optimal arms
        arr = [[0.4, 0.75], [0.75, 0.4]]
        x = 0.45 + 0.2 ** (1 + np.arange(4))
        y = 0.35 - 0.2 ** (1 + np.arange(4))
        x2 = 0.10 + 0.20 ** (1 + np.arange(4))
        y2 = 0.70 - 0.20 ** (1 + np.arange(4))
        z = np.hstack((x[:, None], y[:, None]))
        z2 = np.hstack((x2[:, None], y2[:, None]))
        arr.extend(z)
        arr.extend(z2)
        return np.array(arr)
    elif ds=='I11':
        x = 0.75 - 0.55 ** (1 + np.arange(5))
        return np.array([x, x]).T
    elif ds=="I22":
        x = 0.5 - 0.025 * (1 + np.arange(15))
        return np.array([x, x]).T
    elif ds=="I4":
        # large number of arms
        max_H = 60000 # because we run it for T=H
        H = 1e12
        pi = np.pi
        theta1 = np.linspace(pi / 12, np.pi / 2 - pi / 12, 20)
        theta2 = np.linspace(np.pi / 2 + pi / 6, 2 * np.pi - pi / 6, 180)
        v1 = np.hstack([np.cos(theta1)[:, None], np.sin(theta1)[:, None]])
        v2 = np.hstack([np.cos(theta2)[:, None], np.sin(theta2)[:, None]])
        return np.concatenate((v1, v2))
        means = np.array([])
        while H>max_H:
            means = np.random.uniform(0.15, 0.85, size=(200, 2))
            bandit = Bernoulli(means)
            H = bandit.H
        return means
    elif ds=="I1":
        # 2 clusters of points
        max_H = 60000
        H = 1e12
        means = np.array([])
        while H>max_H:
            means = np.concatenate([np.random.uniform(0.2, 0.4, size=(10, 2)),
                               np.random.uniform(0.5, 0.7, size=(10, 2))])
            bandit = Bernoulli(means)
            H = bandit.H
        return means
    elif ds=="I2":
        # 3 clusters
        # 2 clusters of points
        max_H = 60000
        H = 1e12
        means = np.array([])
        while H > max_H:
            ans = np.concatenate([np.random.uniform(0.2, 0.4, size=(10, 2)),
                                  np.random.uniform(0.5, 0.7, size=(10, 2))])
            x = np.random.uniform(0.2, 0.4, size=(10))
            y = np.random.uniform(0.6, 0.8, size=(10))
            means = np.concatenate((ans, np.array([x, y]).T))
            bandit = Bernoulli(means)
            H = bandit.H
        return means
    elif ds=="I555":
        arr= np.empty((30, 10), float)
        for i in range(10):
            arr[i] = np.full((10),0.4-0.25**(i+1))
            arr[i+10] = np.full((10), 0.4-0.2**(i+1))
        for i in range(10):
            x = np.full((10), 0.5-0.35**(i+1))
            x[i] = 0.5
            arr[i+20] = x
        print(arr.shape)
        return arr
    elif ds=="convex":
        x = np.linspace(0.5, 1, 20)
        y = np.sin(2 * x)
        f1 = lambda x: x ** 2
        f2 = lambda x: 1 / x ** 2 / 4
        xx = np.random.uniform(0.1, 0.5, size=(500, 2))
        qq = (np.prod(xx, -1) < (1 / 5)).nonzero()[0][:30]
        ww = xx[qq]
        return np.concatenate((np.array([f1(x), f2(x)]).T, ww))
    elif ds == "I7":
        arr = []
        c = 0.1
        for i in range(1, 9):
            arr += [[0.3 + (i - 1) * c, 0.8 - (i - 1) * c]]
        for i in range(9, 16):
            arr += [[0.25 + (i - 9) * c, 0.7 - (i - 9) * c], [0.25 + (i - 9) * c, 0.7 - (i - 9) * c - 0.05]]
        return np.array(arr)
    elif ds =="I5":
        # 3 large dimension
        max_H = 60000
        H = 1e12
        means = np.array([])
        while H>max_H:
            means = np.concatenate([np.random.uniform(0.2, 0.45, size=(30, 10)),
                               np.random.uniform(0.5, 0.75, size=(20, 10))])
            bandit = Bernoulli(means)
            H = bandit.H
        return means
    elif ds=="I6":
        x = 0.75 - 0.65**np.arange(1, 11)
        y = 0.25 + 0.65**np.arange(1, 11)
        return np.concatenate(([x], [y])).T

    return np.array([])
