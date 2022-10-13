import numpy as np


def sdof_free_vibr(x, t):
    return [x[1], -0.1 * x[1] - x[0]]


def sdof_harmonic(x, t, forcing_freq=1.2):
    return [x[1], -0.1 * x[1] - x[0] + np.cos(forcing_freq * t)]


def sdof_friction(x, t, friction_force_ratio=0.5, forcing_freq=1.2):
    return [
        x[1],
        -0.1 * x[1]
        - x[0]
        - friction_force_ratio * np.sign(x[1])
        + np.cos(forcing_freq * t),
    ]
