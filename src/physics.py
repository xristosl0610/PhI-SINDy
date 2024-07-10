import numpy as np
import torch as T
from scipy.integrate import solve_ivp
from src.config_data_class import Config
from typing import Tuple


def apply_forcing(
    times: np.ndarray,
    params: Config
) -> np.ndarray:
    """
    Returns the forcing term for the given time instances based on the specified parameters.

    Args:
        times (np.ndarray): A 1D array containing the discrete time values.
        params (Parameters): The parameters of the run, including the type of forcing and related parameters.

    Returns:
        np.ndarray: A 1D array containing the forcing term for the input discrete time instances.

    Raises:
        ValueError: If an invalid `force_flag` is provided in the `params`.
    """

    if params.physics.force_flag == 'Jonswap':
        return np.expand_dims((np.expand_dims(params.physics.aps, 1)
                               * np.cos((np.expand_dims(params.physics.omegaps, 1) * np.expand_dims(times, 0)
                                         + np.expand_dims(params.physics.eps, 1)))).sum(axis=0), axis=1)
    elif params.physics.force_flag == 'mono':
        return params.physics.F0 * np.cos(params.physics.forcing_freq * times)
    else:
        raise ValueError("Invalid excitation flag provided. Valid options are 'Jonswap' or 'mono'.")


def apply_known_physics(x, params) -> (np.ndarray | T.Tensor):
    """
    A function that returns the known part (terms) of the governing equation

    Parameters
    ----------
    x : numpy.ndarray or torch.Tensor
        a 2D array/tensor containing the displacement in the first column and the velocity in the second one
    params : parameters dataclass
        the parameters of the run
    Returns
    -------
    numpy.ndarray or torch.Tensor
        an 1D array/tensor that contains known part of the governing equation
    """
    return - 2 * params.zeta * params.omega * x[:, 1] - params.omega ** 2 * x[:, 0]


def build_true_model(x, t, params):
    """
    A function that gets the displacement, velocity and time as an input, and returns the true vector field output (velocity and acceleration)

    Parameters
    ----------
    x : numpy.ndarray
        a 2D array containing the displacement in the first column and the velocity in the second one
    t : numpy.ndarray
        an 1D array containing the discrete time values
    params : parameters dataclass
        the parameters of the run
    Returns
    -------
    numpy.ndarray
        a 2D array with the two vector field values, velocity as first column and acceleration as second, for the given input x and t
    """

    if params.friction_model == "C":
        friction_force = params.friction_force_ratio
    elif params.friction_model == "DR":
        friction_force = params.friction_force_ratio \
                         + params.DR["a"] * np.log((np.abs(x[1]) + params.DR["eps"]) / params.DR["V_star"]) \
                         + params.DR["b"] * np.log(params.DR["c"] + params.DR["V_star"] / (np.abs(x[1]) + params.DR["eps"]))
    elif params.friction_model is None:
        friction_force = 0

    forcing = apply_forcing(t, params)
    # Check if there is sticking - Zero velocity and friction force bigger than the opposing forces
    if (np.abs(x[1]) < 1e-5) and (np.abs(forcing - params.stiffness * x[0]) <= np.abs(friction_force) * params.F0):
        return np.array([0., 0.])
    else:
        return np.array([x[1],
                         - 2 * params.zeta * params.omega * x[1]
                         - params.omega ** 2 * x[0]
                         - friction_force * params.F0 / params.mass * np.sign(x[1])
                         + forcing / params.mass], dtype=object)


def generate_data(
    params: Config
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves the system's equation and generates the ground truth data based on the defined run parameters.

    Args:
        params (Parameters): The parameters of the run, which define the system and its equations.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - A 1D numpy array with the discrete time instances.
            - A 2D numpy array with the ground truth data, where the first column represents displacements and the second column represents velocities.
    """
    # Generate (noisy) measurements - Training Data
    ts = np.arange(0, params.physics.timefinal, params.hyperparams.timestep)

    # Solve the equation
    sol = solve_ivp(
        lambda t, x: build_true_model(x, t, params),
        t_span=[ts[0], ts[-1]], y0=params.physics.x0, t_eval=ts
        )

    return ts, np.transpose(sol.y)


def calculate_jonswap_excitation(params: Config):
    Hs = 10
    Tp = 0.5
    fp = 1 / Tp
    omegap = 2 * np.pi / Tp
    omegas = np.arange(0.1, 50.01, 0.1)

    sigmap = np.where((omegas < omegap), 0.07, 0.09)

    gamma = 3.3
    fs = omegas / (2 * np.pi)
    beta = np.exp(-0.5 * ((fs / fp - 1) / sigmap) ** 2)
    Sigmaf = 0.3125 * Hs ** 2 * Tp * (fs / fp) ** (-5) * np.exp(-1.25 * (fs / fp) ** (-4)) * (1 - 0.287 * np.log(gamma)) * gamma ** beta

    inds = np.arange(omegas.shape[0])
    noOfHarmonics = omegas.shape[0]
    deltaf = fs[1] - fs[0]
    aps = np.sqrt(2 * Sigmaf[inds] * deltaf)
    eps = 2 * np.pi * np.random.rand(noOfHarmonics)

    ts = np.arange(0, params.physics.timefinal, params.hyperparams.timestep)

    eta = (np.expand_dims(aps, 1) * np.cos(
        (2 * np.pi * np.expand_dims(fs[inds], 1) * np.expand_dims(ts, 0) + np.expand_dims(eps, 1)))).sum(axis=0)

    params.physics.F0 = np.abs(eta).max()
    params.physics.aps = aps
    params.physics.eps = eps
    params.physics.omegaps = omegas[inds]

    # _, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 6))
    # ax1.plot(fs, Sigmaf)
    #
    # ax1.set_ylabel(r"Spectral Density [m$^2$/Hz]", fontsize=24)
    # ax1.set_xlabel(r"Frequency, $f$ [Hz]", fontsize=24)
    # ax2.plot(ts, eta)
    # ax2.set_xlabel(r"$t$ [s]", fontsize=24)
    # ax2.set_ylabel(r"Displacement [m]", fontsize=24)
    #
    # ax1.set_xticks(np.arange(0, 8.1, 1), np.arange(0, 8.1, 1), fontsize=20)
    # ax1.set_yticks(np.arange(0, 10.1, 2), np.arange(0, 10.1, 2), fontsize=20)
    # ax2.set_xticks(np.arange(0, 5.1, 1), np.arange(0, 5.1, 1), fontsize=20)
    # ax2.set_yticks(np.arange(-4, 4.1, 2), np.arange(-4, 4.1, 2), fontsize=20)
    #
    # ax1.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    # ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    # ax1.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    # ax2.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    #
    # plt.show()


def contaminate_measurements(params: Config, x_denoised):
    x = np.random.normal(loc=x_denoised, scale=params.physics.noise_level * np.abs(x_denoised), size=x_denoised.shape)

    # Contaminate omega and zeta with noise
    if params.physics.noisy_input_flag:
        params.physics.omega = np.clip(np.random.normal(loc=params.physics.true_omega, scale=params.physics.omega_noise * params.physics.true_omega), a_max=None, a_min=0.1)
        params.physics.zeta = np.clip(np.random.normal(loc=params.physics.true_zeta, scale=params.physics.zeta_noise * params.physics.true_zeta), a_max=0.99, a_min=0.01)
    else:
        params.physics.omega = params.physics.true_omega
        params.physics.zeta = params.physics.true_zeta

    return x