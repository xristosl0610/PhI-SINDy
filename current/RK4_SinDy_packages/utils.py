import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

## Normalize data
class normalized_data(nn.Module):
    def __init__(self, x):
        super(normalized_data, self).__init__()
        self.x = x
        self.mean = [np.mean(self.x[:,:,i]) for i in range(self.x.shape[2])]
        self.std = [np.std(self.x[:,:,i]) for i in range(self.x.shape[2])]
        
    def normalize_meanstd(self):
        x_nor = np.zeros_like(self.x)
        for i in range(self.x.shape[2]):
            x_nor[:,:,i] = (self.x[:,:,i] - self.mean[i])/self.std[i]
        return x_nor
        

def print_original_noise_signal(x_original,x_noise):

    plot_kws = dict(linewidth=2)
    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    axs.plot(x_noise[0,:, 0], x_noise[0,:, 1], "ko", label="$Noisy data$")
    axs.plot(x_original[0,:, 0], x_original[0,:, 1], "r", label="$Clean data$", **plot_kws)
    axs.legend()
    axs.set(xlabel="$x_1$", ylabel="$x_2$")


def printing_learned_model(Learned_Coeffs,features_name):
    for j in range(Learned_Coeffs.shape[1]):
        str_eqn = 'dx{}\'= '.format(j)
        for i in range(Learned_Coeffs.shape[0]):
            if np.abs(Learned_Coeffs[i,j]) > 1e-10:
                str_eqn = str_eqn.__add__('{0:.3f}'.format(Learned_Coeffs[i,j]) + ' '+  features_name[i] + ' + ')
        print(str_eqn[:-2])

        
def printing_learned_rational_model(Learned_Coeffs1,Learned_Coeffs2,features_name):

    for j in range(Learned_Coeffs1.shape[1]):
        str_eqn = 'dx{}\'= '.format(j)
        for i in range(Learned_Coeffs1.shape[0]):
            if np.abs(Learned_Coeffs1[i,j]) > 1e-10:
                str_eqn = str_eqn.__add__('{0:.3f}'.format(Learned_Coeffs1[i,j]) + ' '+  features_name[i] + ' + ')
        str_eqn = str_eqn[:-2] 
        str_eqn = str_eqn.__add__('// 1.000 + ')
        
        for i in range(Learned_Coeffs2.shape[0]):
            if np.abs(Learned_Coeffs2[i,j]) > 1e-10:
                str_eqn = str_eqn.__add__('{0:.3f}'.format(Learned_Coeffs2[i,j]) + ' '+  features_name[i+1] + ' + ')
        str_eqn = str_eqn[:-2] 
    
        print(str_eqn)
        


