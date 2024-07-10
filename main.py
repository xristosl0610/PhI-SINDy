import os
import numpy as np
import torch as T
from src.config_data_class import Config
from src.physics import apply_forcing, generate_data, contaminate_measurements
from src.phi_sindy import apply_features, CoeffsDictionary, learn_sparse_model
from src.utils import update_params_from_toml


if __name__ == "__main__":

    # CONSTANTS
    print(os.getcwd())
    CONFIG_OVERWRITE = os.path.join(os.getcwd(), 'src', 'config_overwrite.toml')

    # LOAD PARAMS
    params = Config()

    print()
    # params = update_params_from_toml(params, CONFIG_OVERWRITE)

    # RUN EXCITATION

    # GENERATE DATA
    ts, x_denoised = generate_data(params)

    forcing = apply_forcing(ts, params)
    forcing_m = apply_forcing(0.5 * (ts[:-1] + ts[1:]), params)

    if len(forcing.shape) == 1:
        forcing = np.expand_dims(forcing, axis=1)
        forcing_m = np.expand_dims(forcing_m, axis=1)

    if params.physics.noisy_measure_flag:
        x = contaminate_measurements(params, x_denoised)
    else:
        x = x_denoised

    # LEARN SPARSE SOLUTION
    if params.hyperparams.scaling:
        params.physics.mus = T.tensor(np.mean(x, axis=0)).float().unsqueeze(0)
        params.physics.stds = T.tensor(np.std(x, axis=0)).float().unsqueeze(0)

    # Learn the coefficients
    train_dset = T.tensor(x).float()
    times = T.tensor(ts).unsqueeze(1).float()

    no_of_terms = apply_features(train_dset[:2], times[:2], params=params).shape[1]

    coeffs = CoeffsDictionary(no_of_terms)

    coeffs, loss_track = learn_sparse_model(coeffs, train_dset, times, T.tensor(forcing).float(), T.tensor(forcing_m).float(), Params,
                                            lr_reduction=10)
    learnt_coeffs = coeffs.linear.weight.detach().clone().t().numpy().astype(np.float64)
    """
    if Params.scaling:
        Params.mus = T.tensor(np.mean(x, axis=0)).float().unsqueeze(0)
        Params.stds = T.tensor(np.std(x, axis=0)).float().unsqueeze(0)

    # Learn the coefficients
    train_dset = T.tensor(x).float()
    times = T.tensor(ts).unsqueeze(1).float()

    no_of_terms = apply_features(train_dset[:2], times[:2], params=Params).shape[1]

    coeffs = CoeffsDictionary(no_of_terms)

    # Learning Coefficients
    coeffs, loss_track = learn_sparse_model(coeffs, train_dset, times, T.tensor(forcing).float(), T.tensor(forcing_m).float(), Params, lr_reduction=10)
    learnt_coeffs = coeffs.linear.weight.detach().clone().t().numpy().astype(np.float64)

    Params.equation = print_learnt_equation(learnt_coeffs, Params)

    print(f"\n\n{'-' * len(Params.equation)}\n{'The learnt equation is:'.center(len(Params.equation))}\n{Params.equation}\n{'-' * len(Params.equation)}")
    
    # PLOT
    """
