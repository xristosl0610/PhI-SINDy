#!/usr/bin/env python
# coding: utf-8

import sys, os
print( os.path.dirname( os.path.abspath('') ) )
sys.path.append( os.path.dirname( os.path.abspath('') ) )


# import numpy as np
# import torch
# import random
# from scipy.integrate import odeint, solve_ivp
# from sklearn.model_selection import train_test_split
# import torch_optimizer as optim_all
# from torch.optim.lr_scheduler import StepLR
# from dataclasses import dataclass

# import sys
# import os
# import tikzplotlib
# import pysindy as ps
# import polynomial_library_torch as  pl_torch
# from utils import *
# from models import *
# from learning_models import *
# from scipy import signal

# torch.manual_seed(42)
# np.random.seed(seed=42)


import numpy as np
import torch
import os
from scipy.integrate import solve_ivp
from dataclasses import dataclass
import matplotlib.pyplot as plt
import Dictionary.polynomial_library_torch as  pl_torch

from Functions.utils import printing_learned_rational_model, normalized_data
from Functions.modules import coeffs_dictionary_rational

from Functions.models import MM_Kinetics
from Functions.learning_models import learning_sparse_model_rational
from scipy import signal

from IPython.utils.io import Tee
from contextlib import closing
import tikzplotlib

torch.manual_seed(42)
np.random.seed(seed=42)

# In[2]:


@dataclass
class parameters:
    bs: int = 2
    num_epochs: int = 3000
    num_iter = 8
    lr: float = 1e-3
    save_model_path: str = './Results/MMkinectics/Noise/'
    weightdecay: float =0e-3
    NumInitial: int = 4
    dim_x: int = 1
    timefinal: float = 8.0
    timestep: float = 5e-2
    noiselevel: float = 0.02
    normalize: bool = False
    denoising: bool = True
    tol_coeffs: float = 1e-2
    poly_order = 4
    tikz_save: bool = False

Params = parameters()

os.makedirs(os.path.dirname(Params.save_model_path), exist_ok=True)


dynModel = MM_Kinetics

ts = np.arange(0,Params.timefinal,Params.timestep)
# Initial condition and simulation time

x = np.zeros((Params.NumInitial,len(ts),Params.dim_x))
Ts = np.zeros((Params.NumInitial,len(ts),1))

xp0 = np.linspace(0.5,2,num = Params.NumInitial)

for i in range(Params.NumInitial):
    # x0 = np.random.rand(Params.dim_x,)
    x0 = np.array(xp0[i]).reshape(-1,)
    sol = solve_ivp(lambda t, x: dynModel(x, t), [ts[0], ts[-1]], x0, t_eval=ts)
    
    x[i] = np.transpose(sol.y)
    Ts[i] = ts.reshape(-1,1)

x_original = x.copy()


fig = plt.figure(figsize=(4, 2.5))
ax = fig.add_subplot(1, 1, 1)
for i in range(Params.NumInitial):
    ax.plot(Ts[i],x[i],'o', markersize=1)
ax.set(xlabel="time", ylabel="$s(t)$")

tikzplotlib.save(Params.save_model_path + "MMKinectics_InitialCond.tex")
plt.show()
fig.savefig(Params.save_model_path + "MMKinectics_InitialCond.pdf", bbox_inches = 'tight',pad_inches = 0)



####### Adding noise
x = x + Params.noiselevel * np.random.randn(*x.shape)

x_noise = x.copy()
    

x_denoise = np.zeros_like(x_noise)

for i in range(x.shape[0]):
    x_denoise[i,:,0] = signal.savgol_filter(x_noise[i,:,0], 31,3)

    
fig = plt.figure(figsize=(4, 2.5))
ax = fig.add_subplot(111)
for i in range(x.shape[0]):
    if i == 0:
        ax.plot(ts,x_noise[i,:,0],'o',markersize=2)
        ax.plot(ts,x_original[i,:,0],'k--', linewidth = 0.2)
    else:
        ax.plot(ts,x_noise[i,:,0],'o',markersize=2)
        ax.plot(ts,x_original[i,:,0],'k--', linewidth = 0.2)
        
ax.set(xlabel = "time",ylabel="$s$")

if Params.tikz_save:
    tikzplotlib.save(Params.save_model_path + "MM_Kinectics_noisedata.tex")
plt.show()
fig.savefig(Params.save_model_path + "MM_Kinectics_noisedata.pdf", bbox_inches = 'tight',pad_inches = 0)

fig = plt.figure(figsize=(4, 2.5))
ax = fig.add_subplot(111)
for i in range(x.shape[0]):
    if i == 0:
        ax.plot(ts,x_denoise[i,:,0])
    else:
        ax.plot(ts,x_denoise[i,:,0])
plt.gca().set_prop_cycle(None)

for i in range(x.shape[0]):
    if i == 0:
        ax.plot(ts,x_noise[i,:,0],'o',markersize=2,alpha= 0.25)
    else:
        ax.plot(ts,x_noise[i,:,0],'o',markersize=2,alpha= 0.25)
        
ax.set(xlabel = "time",ylabel="$s$")
ax.legend()

if Params.tikz_save:
    tikzplotlib.save(Params.save_model_path + "MM_Kinectics_denoisedata.tex")
plt.show()
fig.savefig(Params.save_model_path + "MM_Kinectics_denoisedata.pdf", bbox_inches = 'tight',pad_inches = 0)



data_nor = normalized_data(x_denoise)
data_nor.mean, data_nor.std

print('='*50)
print('Mean: {}'.format(data_nor.mean) + 'Std: {}'.format(data_nor.std))
print('='*50)

# data_nor.mean = np.array([0.25])
# data_nor.std = np.array([0.1])
x_denoise = data_nor.normalize_meanstd()

# Define dataloaders
train_dset = list(zip(torch.tensor(x_denoise[:,10:-20,:]).float(),Ts[:,10:-20,:]))
train_dl = torch.utils.data.DataLoader(train_dset, batch_size = Params.bs)
dataloaders = {'train': train_dl}


funs_dictionary = pl_torch.PolynomialLibrary(degree = Params.poly_order)
funs_dictionary.fit(x[0])
funs_dictionary_size = funs_dictionary.transform(x[0]).shape[1]
funs_dictionary_size


Coeffs_rational = coeffs_dictionary_rational(funs_dictionary_size,Params.dim_x)

Coeffs_rational, loss_track = learning_sparse_model_rational(funs_dictionary, Coeffs_rational, dataloaders, Params, 
                                                             lr_reduction = 1.1, quite = True)
    

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
for i in range(Params.num_iter):
    ax.semilogy(loss_track[i], label = 'Number of zero terms: {}'.format(i))
ax.legend()
fig.show()


plot_kws = dict(linewidth=2)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)
for i in range(Params.num_iter):
    ax.semilogy(loss_track[i], label = 'Number of zero terms: {}'.format(i))
ax.legend()

fig = plt.figure(figsize=(4, 2.5))
ax = fig.add_subplot(1, 1, 1)
for i in range(Params.num_iter):
    ax.semilogy(i,loss_track[i,-1],'ko')
ax.semilogy(6,loss_track[6,-1],'go', markersize = 20,fillstyle='none')
ax.set(xlabel="Number of forced zero coefficients ", ylabel="Loss")

tikzplotlib.save(Params.save_model_path + "MMKinectics_noise_Pareto.tex")
plt.show()
fig.savefig(Params.save_model_path + "MMKinectics_noise_Pareto.pdf", bbox_inches = 'tight',pad_inches = 0)

####################
Coeffs_rational = torch.load(Params.save_model_path+'MM_model_coefficients_iter_{}.pkl'.format(6))


Learned_Coeffs_numerator = Coeffs_rational.numerator.weight.detach().clone().t().numpy()
Learned_Coeffs_denominator = Coeffs_rational.denominator.weight.detach().clone().t().numpy()

with closing(Tee(Params.save_model_path + "MM_Kinectics_learnedmodel.log", "a+", channel="stdout")) as outputstream:
    # printing of the learned sparse models in a file
    print('\n')
    print('='*50)
    print('RK4 Inspired Methods Sparse Identification')
    printing_learned_rational_model(Learned_Coeffs_numerator, Learned_Coeffs_denominator, funs_dictionary.get_feature_names())
    print('='*50)
    
    
# Adding one to denominator
Learned_Coeffs_denominator = np.concatenate((np.ones((1,1)),Learned_Coeffs_denominator),axis=0)


# Simulating models
fn = lambda z: (funs_dictionary.transform(np.expand_dims(z, axis=0))@Learned_Coeffs_numerator).reshape(-1,)
fd = lambda z: (funs_dictionary.transform(np.expand_dims(z, axis=0))@Learned_Coeffs_denominator).reshape(-1,)

f1 = lambda z: (fn(z)/fd(z))
learnt_deri = lambda z,t: np.array(f1(z))


x0 = np.array([2.])
x0_nor = (x0  - data_nor.mean)/data_nor.std

ts_refine = np.arange(0,Params.timefinal,1e-2)

sol = solve_ivp(lambda t, x: dynModel(x, t), 
            [ts_refine[0], ts_refine[-1]], x0, t_eval=ts_refine)
# x = sol
x_refine = np.transpose(sol.y).reshape(1,-1,Params.dim_x)


sol_learnt = solve_ivp(lambda t, x: learnt_deri(x, t), 
            [ts_refine[0], ts_refine[-1]], x0_nor, t_eval=ts_refine)
x_learnt = np.transpose(sol_learnt.y).reshape(1,-1,Params.dim_x)

x_learnt = x_learnt*data_nor.std + data_nor.mean
    
xp0 = np.linspace(0.25,4,num = 10)

first_fig = True

plot_kws = dict(linewidth=2)
fig, axs = plt.subplots(1, 1, figsize=(4, 2.5))

for i in range(len(xp0)):
    # x0 = np.random.rand(Params.dim_x,)
    x0 = np.array(xp0[i]).reshape(-1,)
    x0_nor = (x0  - data_nor.mean)/data_nor.std

    ts_refine = np.arange(0,Params.timefinal,1e-2)

    sol = solve_ivp(lambda t, x: dynModel(x, t), 
                [ts_refine[0], ts_refine[-1]], x0, t_eval=ts_refine)
    # x = sol
    x_refine = np.transpose(sol.y).reshape(1,-1,Params.dim_x)
    
    
    sol_learnt = solve_ivp(lambda t, x: learnt_deri(x, t), 
                [ts_refine[0], ts_refine[-1]], x0_nor, t_eval=ts_refine)
    x_learnt = np.transpose(sol_learnt.y).reshape(1,-1,Params.dim_x)
    
    x_learnt = x_learnt*data_nor.std + data_nor.mean
    
    if i == 0:
        axs.plot(ts_refine, x_refine[0,:, 0], "k",markersize=3, label="Ground-truth model", **plot_kws)
        axs.plot(ts_refine, x_learnt[0,:, 0], "go", label="RK4-Sindy",markersize=4,markevery=25, **plot_kws)
    else:
        if x0 > 2.0:
            if first_fig == True:
                axs.plot(ts_refine, x_refine[0,:, 0], "k",markersize=3, label="Ground-truth model", **plot_kws)
                axs.plot(ts_refine, x_learnt[0,:, 0], "mo", markersize=3,label="RK4-Sindy(Outside training)",markevery=25, **plot_kws)
                first_fig = False
            else:
                axs.plot(ts_refine, x_refine[0,:, 0], "k",markersize=3, **plot_kws)
                axs.plot(ts_refine, x_learnt[0,:, 0], "mo", markersize=3,markevery=25, **plot_kws)
        else:
            axs.plot(ts_refine, x_refine[0,:, 0], "k",markersize=3, **plot_kws)
            axs.plot(ts_refine, x_learnt[0,:, 0], "go", markersize=3,markevery=25, **plot_kws)
        
    axs.legend()
    axs.set(xlabel="time", ylabel="$s$")

if Params.tikz_save:
    tikzplotlib.save(Params.save_model_path + "MMKinectics_noise_simulation.tex")
plt.show()
fig.savefig(Params.save_model_path + "MMKinectics_noise_simulation.pdf", bbox_inches = 'tight',pad_inches = 0)


    