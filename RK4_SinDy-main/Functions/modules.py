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


class coeffs_dictionary_rational(nn.Module):
    def __init__(self, n_combinations, n_features):

        """
        Defining the sparse coefficiets and in the forward pass,
        we obtain a ratio of multiplications of features and sparse coefficients.
        ----------
        n_combinations : int: the number of features in dictionary
            DESCRIPTION.
        n_features : int : the number of variables
            DESCRIPTION.

        Returns
        -------
        Product of features multiplied by sparse coefficients.

        """
        super(coeffs_dictionary_rational, self).__init__()
        self.numerator = nn.Linear(n_combinations, n_features, bias=False)
        self.denominator = nn.Linear(n_combinations - 1, n_features, bias=False)

        # Setting weights to zero
        self.numerator.weight = torch.nn.Parameter(
            0 * self.numerator.weight.clone().detach()
        )
        self.denominator.weight = torch.nn.Parameter(
            (0 * self.denominator.weight.clone().detach())
        )

    def forward(self, x):
        N1 = self.numerator(x)
        D1 = self.denominator(x[:, 1:])
        return N1 / (D1 + 1.0)


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
def rk4th_onestep_SparseId_Force(x, library, LibsCoeffs, times, timestep=1e-2):

    d1 = library.transform_torch(x)
    k1 = LibsCoeffs(d1) + torch.column_stack(
        (torch.zeros(size=(times.shape)), torch.cos(2 * times))
    )

    d2 = library.transform_torch(x + 0.5 * timestep * k1)
    k2 = LibsCoeffs(d2) + torch.column_stack(
        (torch.zeros(size=(times.shape)), torch.cos(2 * (times + 0.5 * timestep)))
    )

    d3 = library.transform_torch(x + 0.5 * timestep * k2)
    k3 = LibsCoeffs(d3) + torch.column_stack(
        (torch.zeros(size=(times.shape)), torch.cos(2 * (times + 0.5 * timestep)))
    )

    d4 = library.transform_torch(x + 1.0 * timestep * k3)
    k4 = LibsCoeffs(d4) + torch.column_stack(
        (torch.zeros(size=(times.shape)), torch.cos(2 * (times + timestep)))
    )

    return x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * timestep


## Controlled RK-SINDy model
def rk4th_onestep_SparseId_control(
    x, u, library, LibsCoeffs, Params, t=0, timestep=1e-2
):

    d1 = library.transform_torch(x)
    du1 = d1[:, : 1 + Params.dim_x] * u
    k1 = LibsCoeffs(d1, du1)

    d2 = library.transform_torch(x + 0.5 * timestep * k1)
    du2 = d2[:, : 1 + Params.dim_x] * u
    k2 = LibsCoeffs(d2, du2)

    d3 = library.transform_torch(x + 0.5 * timestep * k2)
    du3 = d3[:, : 1 + Params.dim_x] * u
    k3 = LibsCoeffs(d3, du3)

    d4 = library.transform_torch(x + 1.0 * timestep * k3)
    du4 = d4[:, : 1 + Params.dim_x] * u
    k4 = LibsCoeffs(d4, du4)

    return x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * timestep


## Parametric RK-SINDy model
def rk4th_onestep_SparseId_parameter(x, mu, library, LibsCoeffs, t=0, timestep=1e-2):

    x_mu = torch.cat((x, mu), dim=1)
    d1 = library.transform_torch(x_mu)
    k1 = LibsCoeffs(d1)

    x_mu1 = torch.cat((x + 0.5 * timestep * k1, mu), dim=1)

    d2 = library.transform_torch(x_mu1)
    k2 = LibsCoeffs(d2)

    x_mu2 = torch.cat((x + 0.5 * timestep * k2, mu), dim=1)

    d3 = library.transform_torch(x_mu2)
    k3 = LibsCoeffs(d3)

    x_mu3 = torch.cat((x + 1.0 * timestep * k3, mu), dim=1)

    d4 = library.transform_torch(x_mu3)
    k4 = LibsCoeffs(d4)

    return x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * timestep
