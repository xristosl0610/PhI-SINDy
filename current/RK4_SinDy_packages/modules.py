import torch
import torch.nn as nn

## Define coeffis for a dictionary
class coeffs_dictionary(nn.Module):
    def __init__(self, n_combinations, n_features):

        """
        Defining the sparse coefficiets and in the forward pass,
        we obtain multiplication of features and sparse coefficients.
        ----------
        n_combinations : int: the number of features in dictionary
            DESCRIPTION.
        n_features : int : the number of variables
            DESCRIPTION.

        Returns
        -------
        Product of features multiplied by sparse coefficients.

        """
        super(coeffs_dictionary, self).__init__()
        self.linear = nn.Linear(n_combinations, n_features, bias=False)
        # Setting the weights to zeros
        self.linear.weight = torch.nn.Parameter(0 * self.linear.weight.clone().detach())

    def forward(self, x):
        return self.linear(x)


## Simple RK model
def rk4th_onestep(model, x, t=0, timestep=1e-2):
    k1 = model(x, t)
    k2 = model(x + 0.5 * timestep * k1, t + 0.5 * timestep)
    k3 = model(x + 0.5 * timestep * k2, t + 0.5 * timestep)
    k4 = model(x + 1.0 * timestep * k3, t + 1.0 * timestep)
    return x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * timestep


## Simple RK-SINDy model
def rk4th_onestep_SparseId(x, library, LibsCoeffs, t=0, timestep=1e-2):

    d1 = library.transform_torch(x)
    k1 = LibsCoeffs(d1)

    d2 = library.transform_torch(x + 0.5 * timestep * k1)
    k2 = LibsCoeffs(d2)

    d3 = library.transform_torch(x + 0.5 * timestep * k2)
    k3 = LibsCoeffs(d3)

    d4 = library.transform_torch(x + 1.0 * timestep * k3)
    k4 = LibsCoeffs(d4)

    return x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * timestep


## Simple RK-SINDy model with forcing term
def rk4th_onestep_SparseId_Force(
    x, library, LibsCoeffs, times, timestep=1e-2, forcing_freq=1.2
):

    d1 = library.transform_torch(x)
    k1 = LibsCoeffs(d1) + torch.column_stack(
        (torch.zeros(size=(times.shape)), torch.cos(forcing_freq * times))
    )

    d2 = library.transform_torch(x + 0.5 * timestep * k1)
    k2 = LibsCoeffs(d2) + torch.column_stack(
        (
            torch.zeros(size=(times.shape)),
            torch.cos(forcing_freq * (times + 0.5 * timestep)),
        )
    )

    d3 = library.transform_torch(x + 0.5 * timestep * k2)
    k3 = LibsCoeffs(d3) + torch.column_stack(
        (
            torch.zeros(size=(times.shape)),
            torch.cos(forcing_freq * (times + 0.5 * timestep)),
        )
    )

    d4 = library.transform_torch(x + 1.0 * timestep * k3)
    k4 = LibsCoeffs(d4) + torch.column_stack(
        (torch.zeros(size=(times.shape)), torch.cos(forcing_freq * (times + timestep)))
    )

    return x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * timestep


## Simple RK-SINDy model with friction and forcing terms
def rk4th_onestep_SparseId_Force_and_Friction(
    x, library, LibsCoeffs, times, timestep=1e-2, friction_ratio=0.5, forcing_freq=1.2
):

    d1 = library.transform_torch(x)
    k1 = LibsCoeffs(d1) + torch.column_stack(
        (
            torch.zeros(size=(times.shape)),
            -friction_ratio * torch.sign(x[:, 1]).unsqueeze(1)
            + torch.cos(forcing_freq * times),
        )
    )

    xtemp = x + 0.5 * timestep * k1
    d2 = library.transform_torch(xtemp)
    k2 = LibsCoeffs(d2) + torch.column_stack(
        (
            torch.zeros(size=(times.shape)),
            -friction_ratio * torch.sign(xtemp[:, 1]).unsqueeze(1)
            + torch.cos(forcing_freq * (times + 0.5 * timestep)),
        )
    )

    xtemp = x + 0.5 * timestep * k2
    d3 = library.transform_torch(xtemp)
    k3 = LibsCoeffs(d3) + torch.column_stack(
        (
            torch.zeros(size=(times.shape)),
            -friction_ratio * torch.sign(xtemp[:, 1]).unsqueeze(1)
            + torch.cos(forcing_freq * (times + 0.5 * timestep)),
        )
    )

    xtemp = x + timestep * k3
    d4 = library.transform_torch(xtemp)
    k4 = LibsCoeffs(d4) + torch.column_stack(
        (
            torch.zeros(size=(times.shape)),
            -friction_ratio * torch.sign(xtemp[:, 1]).unsqueeze(1)
            + torch.cos(forcing_freq * (times + timestep)),
        )
    )

    return x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * timestep
