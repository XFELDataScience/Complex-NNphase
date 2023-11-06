# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 19:08:26 2022

@author: xyu1
"""

import numpy as np
import math
import cmath

import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader



from torch.nn import Module, Parameter, init
from torch.nn import Conv2d, Linear, BatchNorm1d, BatchNorm2d
from torch.nn import ConvTranspose2d
from torch.nn.functional import relu, max_pool2d, avg_pool2d, dropout, dropout2d, interpolate, sigmoid, tanh

class CConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        
        super(CConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        
        self.re_conv = nn.Conv2d(self.in_channels, self.out_channels, **kwargs)
        self.im_conv = nn.Conv2d(self.in_channels, self.out_channels, **kwargs)
        
        nn.init.xavier_uniform_(self.re_conv.weight)
        nn.init.xavier_uniform_(self.im_conv.weight)

    def forward(self, x):
        
        
        x_re = x[..., 0]
        x_im = x[..., 1]
        
        out_re = self.re_conv(x_re) - self.im_conv(x_im)
        out_im = self.re_conv(x_im) + self.im_conv(x_re)
        
        out = torch.stack([out_re, out_im], -1) 
        
        return out

class CConvTrans2d(nn.Module):
  def __init__(self, in_channels, out_channels, **kwargs):
    super(CConvTrans2d, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels


    self.re_Tconv = nn.ConvTranspose2d(self.in_channels, self.out_channels, **kwargs)
    self.im_Tconv = nn.ConvTranspose2d(self.in_channels, self.out_channels, **kwargs)

    nn.init.xavier_uniform_(self.re_Tconv.weight)
    nn.init.xavier_uniform_(self.im_Tconv.weight)


  def forward(self, x):  
    x_re = x[..., 0]
    x_im = x[..., 1]

    out_re = self.re_Tconv(x_re) - self.im_Tconv(x_im)
    out_im = self.re_Tconv(x_im) + self.im_Tconv(x_re)

    out = torch.stack([out_re, out_im], -1) 

    return out


class CBatchnorm(nn.Module):
    def __init__(self, in_channels):
        super(CBatchnorm, self).__init__()
        self.in_channels = in_channels

        self.re_batch = nn.BatchNorm2d(in_channels)
        self.im_batch = nn.BatchNorm2d(in_channels)


    def forward(self, x):
        x_re = x[..., 0]
        x_im = x[..., 1]

        out_re =  self.re_batch(x_re)
        out_im =  self.re_batch(x_im)


        out = torch.stack([out_re, out_im], -1) 

        return out
    


class CMaxPool2d(nn.Module):
    
    def __init__(self, kernel_size, **kwargs):
        
        
        super(CMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
    
    
        self.CMax_re = nn.MaxPool2d(self.kernel_size, **kwargs)
        self.CMax_im = nn.MaxPool2d(self.kernel_size, **kwargs) 
  
    def forward(self, x):
        
        
        x_re = x[..., 0]
        x_im = x[..., 1]

        out_re = self.CMax_re(x_re)
        out_im = self.CMax_im(x_im)


        out = torch.stack([out_re, out_im], -1) 

        return out
    
def complex_max_pool2d(x ,kernel_size, stride=None, padding=0,
                                dilation=1, ceil_mode=False, return_indices=False):
        
        x_re = x[..., 0]
        x_im = x[..., 1]
        
        x_amp = torch.sqrt(torch.square(x_re) + torch.square(x_im))
        
        _, indices =  max_pool2d(  x_amp, 
                                   kernel_size = kernel_size, 
                                   stride = stride, 
                                   padding = padding, 
                                   dilation = dilation,
                                   ceil_mode = ceil_mode, 
                                   return_indices = True
                                )
        
        out_re = _retrieve_elements_from_indices(x_re, indices)
        
        out_im = _retrieve_elements_from_indices(x_im, indices)
        
        
        out = torch.stack([out_re, out_im], -1) 
        
        return out
        
def complex_upsample(x, size=None, scale_factor=None, mode='nearest',
                             align_corners=None, recompute_scale_factor=None):
    '''
        Performs upsampling by separately interpolating the real and imaginary part and recombining
    '''
    
    x_re = x[..., 0]
    x_im = x[..., 1]
    
    outp_real = interpolate(x_re,  size=size, scale_factor=scale_factor, mode=mode,
                                    align_corners=align_corners, recompute_scale_factor=recompute_scale_factor)
    outp_imag = interpolate(x_im,  size=size, scale_factor=scale_factor, mode=mode,
                                    align_corners=align_corners, recompute_scale_factor=recompute_scale_factor)
    
    
    out = torch.stack([outp_real, outp_imag], -1) 
    
    return out





        
        
        
        
        
        
        
        
        


        
  
  
  













