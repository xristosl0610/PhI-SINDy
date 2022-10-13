import numpy as np
import scipy.integrate as spint


def sdof_free_vibr(x, t):
    return [x[1], -0.1 * x[1] - x[0]]


def sdof_harmonic(x, t):
    return [x[1], -0.1 * x[1] - x[0] + np.cos(2 * t)]


def lorenz_model(x, t):
    return [
        10 * (x[1] - x[0]),
        x[0] * (28 - x[2]) - x[1],
        x[0] * x[1] - 8 / 3 * x[2],
    ]


def MM_Kinetics(x, t):
    return [((0.6 - 3 * x) / (1 + (10 * x / 3)))]


def FHN_model(x, t):
    a, b, g, I = (0.8, 0.7, 1 / 25, 0.5)
    dx1 = x[0] - (x[0] ** 3) / 3 - x[1] + I
    dx2 = g * (x[0] + a - b * x[1])
    return np.array([dx1, dx2])


def linear_2D(x, t):
    return [-0.1 * x[0] + 2 * x[1], -2 * x[0] - 0.1 * x[1]]


def cubic_2D(x, t):
    return [
        -0.1 * x[0] ** 3 + 2 * x[1] ** 3,
        -2 * x[0] ** 3 - 0.1 * x[1] ** 3,
    ]


def hopf(x, mu, omega, A):
    return [
        mu * x[0] - omega * x[1] - A * x[0] * (x[0] ** 2 + x[1] ** 2),
        omega * x[0] + mu * x[1] - A * x[1] * (x[0] ** 2 + x[1] ** 2),
    ]


class Compute_Mean_Std:
    def __init__(self, data, print_opts=False):
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
        if print_opts:
            print("_" * 30)
            print("mean: {}".format(self.mean))
            print("std: {}".format(self.std))
