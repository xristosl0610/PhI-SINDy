""" Training of a network """
import torch
import sys
import torch_optimizer as optim_all
import numpy as np
from modules import (
    rk4th_onestep_SparseId,
    rk4th_onestep_SparseId_Force,
    rk4th_onestep_SparseId_Force_and_Friction,
)


def learning_sparse_model(
    dictionary,
    Coeffs,
    train_set,
    times,
    Params,
    lr_reduction=10,
    quite=False,
):

    """
    Parameters
    ----------
    dictionary : A function
        It is a symbolic dictionary, containing potential candidate functions that describes dynamics.
    Coeffs : float
        Coefficients that picks correct features from the dictionary .
    train_set : tensor
        Containing training data that follows PyTorch framework.
    times : tensor
        Containing the discrete time values corresponfing to the collected training data
    Params : dataclass
        Containing additional auxilary parameters.
    lr_reduction : float, optional
        The learning rate is reduced by lr_reduction after each iteration. The default is 10.
    quite : bool, optional
        It decides whether to print coeffs after each iteration. The default is False.

    Returns
    -------
    Coeffs : float
        Non-zero coefficients picks features from the dictionary and
        also determines right coefficients in front of the features.
    loss_track : float
        tacking loss after each epoch and iteration.

    """

    # Define optimizer
    opt_func = optim_all.RAdam(
        Coeffs.parameters(), lr=Params.lr, weight_decay=Params.weightdecay
    )
    # Define loss function
    criteria = torch.nn.MSELoss()
    # pre-allocate memory for loss_fuction
    loss_track = np.zeros((Params.num_iter, Params.num_epochs))
    #########################
    ###### Training #########
    #########################
    for p in range(Params.num_iter):
        for g in range(Params.num_epochs):
            Coeffs.train()

            opt_func.zero_grad()

            loss_new = torch.autograd.Variable(torch.tensor([0.0], requires_grad=True))
            weights = 2 ** (-0.5 * torch.linspace(0, 0, 1))

            timesteps_i = torch.tensor(np.diff(times, axis=0)).float()
            # times = torch.tensor(y[1][i]).float()
            y_total = train_set

            if Params.model == "free":
                ##################################
                # One forward step predictions
                ##################################
                y_pred = rk4th_onestep_SparseId(
                    y_total[:-1], dictionary, Coeffs, timestep=timesteps_i
                )

                ##################################
                # One backward step predictions
                ##################################
                y_pred_back = rk4th_onestep_SparseId(
                    y_total[1:], dictionary, Coeffs, timestep=-timesteps_i
                )
            elif Params.model == "forced":
                ##################################
                # One forward step predictions
                ##################################
                y_pred = rk4th_onestep_SparseId_Force(
                    y_total[:-1],
                    dictionary,
                    Coeffs,
                    times[:-1],
                    timestep=timesteps_i,
                    forcing_freq=Params.forcing_freq,
                )

                ##################################
                # One backward step predictions
                ##################################
                y_pred_back = rk4th_onestep_SparseId_Force(
                    y_total[1:],
                    dictionary,
                    Coeffs,
                    times[1:],
                    timestep=-timesteps_i,
                    forcing_freq=Params.forcing_freq,
                )
            elif Params.model == "friction":
                ##################################
                # One forward step predictions
                ##################################
                y_pred = rk4th_onestep_SparseId_Force_and_Friction(
                    y_total[:-1],
                    dictionary,
                    Coeffs,
                    times[:-1],
                    timestep=timesteps_i,
                    friction_ratio=Params.friction_ratio,
                    forcing_freq=Params.forcing_freq,
                )

                ##################################
                # One backward step predictions
                ##################################
                y_pred_back = rk4th_onestep_SparseId_Force_and_Friction(
                    y_total[1:],
                    dictionary,
                    Coeffs,
                    times[1:],
                    timestep=-timesteps_i,
                    friction_ratio=Params.friction_ratio,
                    forcing_freq=Params.forcing_freq,
                )

            loss_new += criteria(y_pred, y_total[1:]) + weights[0] * criteria(
                y_pred_back, y_total[:-1]
            )

            # loss_new /= y[0].shape[0]
            loss_track[p, g] += loss_new.item()
            loss_new.backward()
            opt_func.step()

            sys.stdout.write(
                "\r [Iter %d/%d] [Epoch %d/%d] [Training loss: %.2e] [Learning rate: %.2e]"
                % (
                    p + 1,
                    Params.num_iter,
                    g + 1,
                    Params.num_epochs,
                    loss_track[p, g],
                    opt_func.param_groups[0]["lr"],
                )
            )

        # Removing the coefficients smaller than tol and set gradients w.r.t. them to zero
        # so that they will not be updated in the iterations
        Ws = Coeffs.linear.weight.detach().clone()
        Mask_Ws = (Ws.abs() > Params.tol_coeffs).type(torch.float)
        Coeffs.linear.weight = torch.nn.Parameter(Ws * Mask_Ws)

        if not quite:
            print("\n")
            print(Ws)
            print(
                "\nError in coeffs due to truncation: {}".format(
                    (Ws - Coeffs.linear.weight).abs().max()
                )
            )
            print("Printing coeffs after {} iter after truncation".format(p + 1))
            print(Coeffs.linear.weight)
            print("\n" + "=" * 50)

        Coeffs.linear.weight.register_hook(lambda grad: grad.mul_(Mask_Ws))
        new_lr = opt_func.param_groups[0]["lr"] / lr_reduction
        opt_func = optim_all.RAdam(
            Coeffs.parameters(), lr=new_lr, weight_decay=Params.weightdecay
        )

    return Coeffs, loss_track
