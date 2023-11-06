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

#from tqdm import tqdm 
import numpy as np
#from skimage.transform import resize
from numpy.fft import fftn, fftshift

import matplotlib.pyplot as plt
import matplotlib
from math import sqrt

import os
import argparse
import glob

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import numpy as np
from tqdm import tqdm 

from torch.nn import Module, Parameter, init
from torch.nn import Conv2d, Linear, BatchNorm1d, BatchNorm2d
from torch.nn import ConvTranspose2d
from torch.nn.functional import relu, max_pool2d, avg_pool2d, dropout, dropout2d, interpolate, sigmoid, tanh
from complexNN import CConv2d, CConvTrans2d, CMaxPool2d


def first_conv(in_channel, out_channel):
    conv = nn.Sequential(CConv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=(1,1)), #One conv to shape how Inception blocks need
            CBatchnorm(64),
            nn.LeakyReLU(0.2),)
    
    return conv
    
def dual_conv(in_channel, out_channel):
    conv = nn.Sequential(
        CConv2d(in_channel, out_channel, kernel_size=3,stride=1, padding=(1,1)),
        nn.LeakyReLU(0.2,inplace= True),
        CConv2d(out_channel, out_channel, kernel_size=3,stride=1, padding=(1,1)),
        nn.LeakyReLU(0.2,inplace= True),
    )
    return conv


def crop_tensor(target_tensor, tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2

    return tensor[:, :, delta:tensor_size- delta, delta:tensor_size-delta]

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        # Left side (contracting path)
        self.dwn_conv1 = dual_conv(1, 32)
        self.dwn_conv2 = dual_conv(32, 64)
        self.dwn_conv3 = dual_conv(64, 128)
        self.dwn_conv4 = dual_conv(128, 256)
        self.maxpool = CMaxPool2d(kernel_size=2, stride=2)

        #Right side  (expnsion path) 
        #transpose convolution is used showna as green arrow in architecture image
        self.trans1 = CConvTrans2d(256,128, kernel_size=2, stride= 2)
        self.up_conv1 = dual_conv(256,128)
        self.trans2= CConvTrans2d(128, 64, kernel_size=2, stride= 2)
        self.up_conv2 = dual_conv(128,64)
        self.trans3 = CConvTrans2d(64, 32, kernel_size=2, stride= 2)
        


        #output layer
        self.out = CConv2d(32, 1, kernel_size=1)

    def forward(self, image):

        #forward pass for Left side
        #crop image first


        x1 = self.dwn_conv1(image)
        x2 = self.maxpool(x1)
        x3 = self.dwn_conv2(x2)
        x4 = self.maxpool(x3)
        x5 = self.dwn_conv3(x4)
        x6 = self.maxpool(x5)
        x7 = self.dwn_conv4(x6)

       
        x = self.trans1(x7)
        y = crop_tensor(x, x5)
        x = self.up_conv1(torch.cat([x,y], 1))

        x = self.trans2(x)
        y = crop_tensor(x, x3)
        x = self.up_conv2(torch.cat([x,y], 1))
        
        
        x = self.trans3(x)

        x = self.out(x)
        
        return x
        
        
        
        
        
        
        
        
        


        
  
  
  













