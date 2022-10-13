""" Training of a network """
import torch
import sys
import torch_optimizer as optim_all
import numpy as np
from .modules import (
    rk4th_onestep_SparseId,
    rk4th_onestep_SparseId_Force,
    rk4th_onestep_SparseId_parameter,
)


def learning_sparse_model(
    dictionary, Coeffs, dataloaders, Params, lr_reduction=10, quite=False
):

    """
    Parameters
    ----------
    dictionary : A function
        It is a symbolic dictionary, containing potential candidate functions that describes dynamics.
    Coeffs : float
        Coefficients that picks correct features from the dictionary .
    dataloaders : dataset
        dataloaders contains the data that follows PyTorch framework.
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
            for y in dataloaders["train"]:
                opt_func.zero_grad()

                loss_new = torch.autograd.Variable(
                    torch.tensor([0.0], requires_grad=True)
                )
                weights = 2 ** (-0.5 * torch.linspace(0, 0, 1))

                for i in range(y[0].shape[0]):
                    yi = y[0][i]
                    timesteps_i = torch.tensor(np.diff(y[1][i], axis=0)).float()
                    print(timesteps_i.shape)
                    # times = torch.tensor(y[1][i]).float()
                    y_total = yi

                    ##################################
                    # One forward step predictions
                    ##################################
                    y_pred = rk4th_onestep_SparseId(
                        y_total[:-1], dictionary, Coeffs, timestep=timesteps_i
                    )
                    # y_pred = rk4th_onestep_SparseId_Force(
                    #     y_total[:-1],
                    #     dictionary,
                    #     Coeffs,
                    #     times[:-1],
                    #     timestep=timesteps_i,
                    # )
                    loss_new += criteria(y_pred, y_total[1:])

                    ##################################
                    # One backward step predictions
                    ##################################
                    y_pred_back = rk4th_onestep_SparseId(
                        y_total[1:], dictionary, Coeffs, timestep=-timesteps_i
                    )
                    # y_pred_back = rk4th_onestep_SparseId_Force(
                    #     y_total[1:],
                    #     dictionary,
                    #     Coeffs,
                    #     times[1:],
                    #     timestep=-timesteps_i,
                    # )

                    loss_new += weights[0] * criteria(y_pred_back, y_total[:-1])

                loss_new /= y[0].shape[0]
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


def learning_sparse_model_parameter(
    dictionary, Coeffs, dataloaders, Params, lr_reduction=10, quite=False
):

    """
    Here, we tailor sparse learning for parameter cases. The script is tested for a single parametes.
    Parameters
    ----------
    dictionary : A function
        It is a symbolic dictionary, containing potential candidate functions that describes dynamics.
    Coeffs : float
        Coefficients that picks correct features from the dictionary .
    dataloaders : dataset
        dataloaders contains the data that follows PyTorch framework.
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
    # Define loss functions
    criteria = torch.nn.MSELoss()
    # pre-allocate memory for loss_fuction
    loss_track = np.zeros((Params.num_iter, Params.num_epochs))

    #########################
    ###### Training #########
    #########################
    for p in range(Params.num_iter):
        for g in range(Params.num_epochs):
            Coeffs.train()
            for y in dataloaders["train"]:
                opt_func.zero_grad()

                loss_new = torch.autograd.Variable(
                    torch.tensor([0.0], requires_grad=True)
                )
                weights = 2 ** (-0.5 * torch.linspace(0, 0, 1))

                for i in range(y[0].shape[0]):
                    yi = y[0][i]
                    mui = y[2][i]
                    timesteps_i = torch.tensor(np.diff(y[1][i], axis=0)).float()

                    ##########################
                    # One forward step predictions
                    y_pred = rk4th_onestep_SparseId_parameter(
                        yi[:-1], mui[:-1], dictionary, Coeffs, timestep=timesteps_i
                    )
                    loss_new += criteria(y_pred, yi[1:])

                    # One backward step predictions
                    y_pred_back = rk4th_onestep_SparseId_parameter(
                        yi[1:], mui[:-1], dictionary, Coeffs, timestep=-timesteps_i
                    )
                    loss_new += weights[0] * criteria(y_pred_back, yi[:-1])

                loss_new /= y[0].shape[0]
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


def learning_sparse_model_rational(
    dictionary, Coeffs_rational, dataloaders, Params, lr_reduction=10, quite=False
):

    """
    Here, we tailor sparse learning for parameter cases. The script is tested for a single parametes.
    Parameters
    ----------
    dictionary : A function
        It is a symbolic dictionary, containing potential candidate functions that describes dynamics.
    Coeffs : float
        Coefficients that picks correct features from the dictionary .
    dataloaders : dataset
        dataloaders contains the data that follows PyTorch framework.
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
        Coeffs_rational.parameters(), lr=Params.lr, weight_decay=Params.weightdecay
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
            Coeffs_rational.train()

            for y in dataloaders["train"]:
                opt_func.zero_grad()

                loss_new = torch.autograd.Variable(
                    torch.tensor([0.0], requires_grad=True)
                )
                weights = 2 ** (-0.5 * torch.linspace(0, 0, 1))

                for i in range(y[0].shape[0]):
                    yi = y[0][i]
                    timesteps_i = torch.tensor(np.diff(y[1][i], axis=0)).float()
                    y_total = yi

                    ##########################
                    # One forward step predictions
                    y_pred = rk4th_onestep_SparseId(
                        y_total[:-1], dictionary, Coeffs_rational, timestep=timesteps_i
                    )
                    loss_new += criteria(y_pred, y_total[1:])

                    # One backward step predictions
                    y_pred_back = rk4th_onestep_SparseId(
                        y_total[1:], dictionary, Coeffs_rational, timestep=-timesteps_i
                    )
                    loss_new += weights[0] * criteria(y_pred_back, y_total[:-1])

                loss_new /= y[0].shape[0]
                loss_track[p, g] += loss_new.item()
                loss_new.backward()
                opt_func.step()

            sys.stdout.write(
                "\r [Forced zero terms %d/%d] [Epoch %d/%d] [Training loss: %.2e] [Learning rate: %.2e]"
                % (
                    p,
                    Params.num_iter,
                    g + 1,
                    Params.num_epochs,
                    loss_track[p, g],
                    opt_func.param_groups[0]["lr"],
                )
            )

        torch.save(
            Coeffs_rational,
            Params.save_model_path + "MM_model_coefficients_iter_{}.pkl".format(p),
        )

        # Removing the coefficients smaller than tol and set gradients w.r.t. them to zero
        # so that they will not be updated in the iterations
        Ws_Num = Coeffs_rational.numerator.weight.detach().clone()
        Ws_Den = Coeffs_rational.denominator.weight.detach().clone()

        if len(Ws_Den[Ws_Den != 0]) == 0:
            Adp_tol = torch.min(Ws_Num[Ws_Num != 0].abs().min()) + 1e-5
        else:
            Adp_tol = (
                torch.min(
                    Ws_Num[Ws_Num != 0].abs().min(), Ws_Den[Ws_Den != 0].abs().min()
                )
                + 1e-5
            )

        Mask_Ws_Num = (Ws_Num.abs() > Adp_tol).type(torch.float)
        Mask_Ws_Den = (Ws_Den.abs() > Adp_tol).type(torch.float)

        Coeffs_rational.numerator.weight = torch.nn.Parameter(Ws_Num * Mask_Ws_Num)
        Coeffs_rational.denominator.weight = torch.nn.Parameter(Ws_Den * Mask_Ws_Den)

        Coeffs_rational.numerator.weight.register_hook(
            lambda grad: grad.mul_(Mask_Ws_Num)
        )
        Coeffs_rational.denominator.weight.register_hook(
            lambda grad: grad.mul_(Mask_Ws_Den)
        )

        new_lr = opt_func.param_groups[0]["lr"] / lr_reduction
        opt_func = optim_all.RAdam(
            Coeffs_rational.parameters(), lr=new_lr, weight_decay=Params.weightdecay
        )

    return Coeffs_rational, loss_track
