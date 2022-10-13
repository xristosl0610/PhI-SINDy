#!/usr/bin/env python
# coding: utf-8

import sys, os
print( os.path.dirname( os.path.abspath('') ) )
sys.path.append( os.path.dirname( os.path.abspath('') ) )


import numpy as np
import torch
import os
from scipy.integrate import solve_ivp
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pysindy as ps
import Dictionary.polynomial_library_torch as  pl_torch
from Functions.utils import printing_learned_model, normalized_data
from Functions.modules import coeffs_dictionary
from Functions.models import lorenz_model
from Functions.learning_models import learning_sparse_model
from IPython.utils.io import Tee
from contextlib import closing
from scipy import signal
import tikzplotlib


torch.manual_seed(42)
np.random.seed(seed=42)

    
@dataclass
class parameters:
    bs: int = 1
    num_epochs: int = 1000
    num_iter = 10  
    lr: float = 1e-1
    save_model_path: str = './Results/Lorenz/'
    weightdecay: float =0.0
    NumInitial: int = 1
    dim_x: int = 3
    timefinal: float = 20.0
    timestep: float = 1e-2
    normalize: bool = True
    tol_coeffs: float = 5e-1
    denoising: bool = True
    poly_order: int = 3
    noiselevel: float = 1.0
    tikz_save: bool = False
    

Params = parameters()

os.makedirs(os.path.dirname(Params.save_model_path), exist_ok=True)

dynModel = lorenz_model

#######################################################################
######## Collecting data ##############################################
#######################################################################
print('='*75 + '\n' + 'Collecting data -- time steping : ' + f'{Params.timestep}\n' + '='*75 + '\n')
ts = np.arange(0,Params.timefinal,Params.timestep)
# Initial condition and simulation time
x0 = [-8, 7, 27]
# Solve the equation
sol = solve_ivp(lambda t, x: dynModel(x, t), [ts[0], ts[-1]], x0, t_eval=ts)
x = np.transpose(sol.y).reshape(1,-1,Params.dim_x)
x_noisefree = x.copy()
    
x = x + Params.noiselevel * np.random.randn(*x.shape)

x_original = x.copy()


x_noise = x.copy()


# Denoising the signal    
if Params.denoising:
    x_denoise = np.zeros_like(x_noise)
    x_denoise[0,:,0] = signal.savgol_filter(x_noise[0,:,0], 15, 3)
    x_denoise[0,:,1] = signal.savgol_filter(x_noise[0,:,1], 15, 3)
    x_denoise[0,:,2] = signal.savgol_filter(x_noise[0,:,2], 15, 3)
else:
    x_denoise = x_noise.copy()
    
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(111)
ax.plot(ts,x_noise[0,:,0],'r',label = 'Noisy-data')
ax.plot(ts,x_noise[0,:,1],'r')
ax.plot(ts,x_noise[0,:,2],'r')
ax.plot(ts,x_denoise[0,:,0],'k--',label = 'Denoised-data')
ax.plot(ts,x_denoise[0,:,1],'k--')
ax.plot(ts,x_denoise[0,:,2],'k--')
ax.set(xlabel = "time",ylabel="$\{x,y,z\}$")
ax.legend()

if Params.tikz_save:
    tikzplotlib.save(Params.save_model_path + "Lorenz_denoisedata.tex")
plt.show()
fig.savefig(Params.save_model_path + "Lorenz_denoisedata.pdf", bbox_inches = 'tight',pad_inches = 0)

# Normalize data
if Params.normalize:
    data_nor = normalized_data(x_denoise)
    data_nor.mean = [0.,0.,25.]
    data_nor.std = [8.,8.,8.]
    x_nor = data_nor.normalize_meanstd()
    
# Define dataloaders
train_dset = list(zip(torch.tensor(x_nor[:,10:-10,:]).float(),ts[10:-10].reshape(1,-1,1)))
train_dl = torch.utils.data.DataLoader(train_dset, batch_size =Params.bs)
dataloaders = {'train': train_dl}

# defining library
funs_dictionary = pl_torch.PolynomialLibrary(degree = Params.poly_order)
funs_dictionary.fit(x[0])
funs_dictionary_size = funs_dictionary.transform(x[0]).shape[1]

Coeffs = coeffs_dictionary(funs_dictionary_size,Params.dim_x)

#######################################################################
######## Applying RK4SINDy ############################################
#######################################################################
# Learning Coefficients
print('='*75 + '\n' + 'Learning sparse coefficients (RK4SINDy)\n' + '='*75 + '\n')
Coeffs, loss_track = learning_sparse_model(funs_dictionary, Coeffs, dataloaders, Params,lr_reduction = 2.0,
                                           quite = True)
Learned_Coeffs = Coeffs.linear.weight.detach().clone().t().numpy()
    
#######################################################################
######## Applying STD-SINDy ############################################
#######################################################################
print('='*75 + '\n' + 'Building models using SINDy\n' + '='*75 + '\n')
# Fit a model using Sindy
threshold = Params.tol_coeffs
deri_method = ps.SINDyDerivative(kind='savitzky_golay', left=0.05, right=0.05, order=3)

model = ps.SINDy(
    optimizer=ps.STLSQ(threshold=threshold),
    differentiation_method=deri_method,
    feature_library=ps.PolynomialLibrary(degree=Params.poly_order)
    )
model.fit(x_nor.reshape(-1,Params.dim_x), t=Params.timestep);
model.print()

#######################################################################
######## Simulating models ############################################
#######################################################################
print('='*75 + '\n' + 'Simulating original and learned models\n' + '='*75 + '\n')
# Simulate all identified model
f1 = lambda z: (funs_dictionary.transform(np.expand_dims(z, axis=0))@Learned_Coeffs).reshape(-1,)
learnt_deri = lambda z,t: np.array(f1(z))

x0 = np.array([x0]).reshape(-1,)

if Params.normalize:
    x0_nor = ((x0 - np.array(data_nor.mean))/data_nor.std).reshape(-1,)
    
ts_refine = np.arange(0,2*Params.timefinal,5e-3)

sol1 = solve_ivp(lambda t, x: dynModel(x, t), 
            [ts_refine[0], ts_refine[-1]], x0, t_eval=ts_refine)
x = np.transpose(sol1.y).reshape(1,-1,Params.dim_x)

sol_learnt = solve_ivp(lambda t, x: learnt_deri(x, t), 
            [ts_refine[0], ts_refine[-1]], x0_nor, t_eval=ts_refine)
x_learnt = np.transpose(sol_learnt.y).reshape(1,-1,Params.dim_x)



x_sindy = model.simulate(x0_nor, ts_refine).reshape(1,-1,Params.dim_x)

if Params.normalize:
       x_learnt = x_learnt*data_nor.std + data_nor.mean
       x_sindy = x_sindy*data_nor.std + data_nor.mean

fig = plt.figure(figsize=(11, 4))
ax = fig.add_subplot(131, projection="3d")
ax.plot(
    x_noise[0,:, 0],
    x_noise[0,:, 1],
    x_noise[0,:, 2], 'ro', markersize=1.2, alpha = 0.6
)
ax.plot(
    x_noisefree[0,:, 0],
    x_noisefree[0,:, 1],
    x_noisefree[0,:, 2], 'k:', linewidth = 0.2,
)
plt.title("Noisy-data")
ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")

ax = fig.add_subplot(132, projection="3d")
ax.plot(
    x_learnt[0,:, 0],
    x_learnt[0,:, 1],
    x_learnt[0,:, 2], 
)
plt.title("RK4-SINDy")
ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")

ax = fig.add_subplot(133, projection="3d")
ax.plot(
    x_sindy[0,:, 0],
    x_sindy[0,:, 1],
    x_sindy[0,:, 2], 'magenta'
)    
plt.title("Std-SINDy")
ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")

if Params.tikz_save:
    tikzplotlib.save(Params.save_model_path + "Lorenz_noise{}.tex".format(Params.noiselevel))
plt.show()
fig.savefig(Params.save_model_path + "Lorenz_noise{}.pdf".format(Params.noiselevel), bbox_inches = 'tight',pad_inches = 0)


plot_kws = dict(linewidth=2)
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].plot(ts, x_original[0,:, 0], "ro",markersize=4, label="$Data$", **plot_kws)
axs[0].plot(ts, x_original[0,:, 1], "ro", markersize=4, **plot_kws)
axs[0].plot(ts_refine, x_learnt[0,:, 0], "k-", label="RK4-SINDy", **plot_kws)
axs[0].plot(ts_refine, x_learnt[0,:, 1], "k-")
axs[0].plot(ts_refine, x_sindy[0,:, 0], "g:", label="SINDy", **plot_kws)
axs[0].plot(ts_refine, x_sindy[0,:, 1], "g:")
axs[0].legend()
axs[0].set(xlabel="t", ylabel="$x_k$")

axs[1].plot(x_original[0,:, 0], x_original[0,:, 1], "ro",markersize=4, label="model", **plot_kws)
axs[1].plot(x_learnt[0,:, 0], x_learnt[0,:, 1], "k-", markersize=4, label="RK4-SINDy", **plot_kws)
axs[1].plot(x_sindy[0,:, 0], x_sindy[0,:, 1], "g:", label="SINDy", **plot_kws)
axs[1].legend()
axs[1].set(xlabel="$x_1$", ylabel="$x_2$")

if Params.tikz_save:
    tikzplotlib.save(Params.save_model_path + "Lorenz_timestep_{}.tex".format(Params.timestep))
plt.show()
fig.savefig(Params.save_model_path + "Lorenz_timestep_{}.pdf".format(Params.timestep), 
            bbox_inches = 'tight',pad_inches = 0)

fig = plt.figure(figsize=(12.5, 4))
ax = fig.add_subplot(131, projection="3d")
ax.plot(
x[0,:, 0],
x[0,:, 1],
x[0,:, 2], "k"
)
ax.plot(
    x_original[0,:, 0],
    x_original[0,:, 1],
    x_original[0,:, 2], "ro", markersize = 1.25,
)
plt.title("Data")
ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")

ax = fig.add_subplot(132, projection="3d")
ax.plot(
    x_learnt[0,:, 0],
    x_learnt[0,:, 1],
    x_learnt[0,:, 2], 
)
plt.title("RK4-SINDy")
ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")

ax = fig.add_subplot(133, projection="3d")
ax.plot(
    x_sindy[0,:, 0],
    x_sindy[0,:, 1],
    x_sindy[0,:, 2],
)    
plt.title("SINDy")
ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")

if Params.tikz_save:
    tikzplotlib.save(Params.save_model_path + "Lorenz_timestep_{}.tex".format(Params.timestep))
plt.show()
fig.savefig(Params.save_model_path + "Lorenz_timestep_{}.pdf".format(Params.timestep), bbox_inches = 'tight',pad_inches = 0)

with closing(Tee(Params.save_model_path + "Lorenz_noisey_DiscoveredModels.log", "a+", channel="stdout")) as outputstream:
    # printing of the learned sparse models in a file
    print('\n')
    print('='*50)
    print('RK4 Inspired Methods Sparse Identification')
    printing_learned_model(Learned_Coeffs,funs_dictionary.get_feature_names())
    print('='*50)
    print('Sindy Approach')
    model.print()
    print('='*50)

