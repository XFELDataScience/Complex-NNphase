# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 19:08:26 2022

@author: xyu1
"""

import numpy as np
import math
import cmath
from matplotlib import pyplot as plt

import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#from torchsummary import summary
from torch.utils.data import TensorDataset, DataLoader
#from pytorch_model_summary import summary

import numpy as np
from numpy.fft import fftn, fftshift
import math
import torch
from torch import nn



def _retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=-2)
    output = flattened_tensor.gather(dim=-1, index=indices.flatten(start_dim=-2)).view_as(indices)
    return output
    

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
        

def Get_Gaussian_support(threshold):
    N2=64
    G2 = np.zeros([N2,N2])
    sigx=9   #X ellipse width in pixels (support)
    sigy=30    #Y ellipse width
    rot=-math.pi*35/180     # rotation angle
    for i in range(N2):
        for j in range(N2):
            x=(i-N2/2)*math.sin(rot)+(j-N2/2)*math.cos(rot)
            y=(i-N2/2)*math.cos(rot)-(j-N2/2)*math.sin(rot)
            G2[i][j] = math.exp(-x**2/sigx**2-y**2/sigy**2) #Gaussian
    
    G_support = np.zeros([N2,N2])
    G_support[np.where(G2>threshold)] = 1
    
    return torch.tensor(G_support)
    







        
        
        
        
        
        
        
        
        


        
  
  
  













