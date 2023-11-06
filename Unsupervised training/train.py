# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 15:47:38 2022

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
from skimage.metrics import structural_similarity as ssim

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
from sklearn.utils import shuffle

from torch.nn import Module, Parameter, init
from torch.nn import Conv2d, Linear, BatchNorm1d, BatchNorm2d
from torch.nn import ConvTranspose2d
from torch.nn.functional import relu, max_pool2d, avg_pool2d, dropout, dropout2d, interpolate, sigmoid, tanh
from scipy import ndimage
from model import Unet
from utils import propcessing_img, real_datasets,get_lr,Get_Gaussian_support


parser = argparse.ArgumentParser(description='PyTorch Phase amplitude retrieval ')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train')




parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--no-pretrain', action='store_true', default=True,
                    help='disables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')


parser.add_argument('--model-dir', default='./trained_model_AE__real_unet',
                    help='directory of model for saving checkpoint')
                    
parser.add_argument('--real-img-path', default='',
                    help='directory of experimental data')
parser.add_argument('--pretrain-real-path', default='',
                    help='directory of real part of experimental data')
parser.add_argument('--pretrain-imagary-path', default='',
                    help='directory of real part of experimental data')

parser.add_argument('--save-dir', default='./trained_model_test_AE',
                    help='directory of model for saving checkpoint')
args = parser.parse_args()

device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')

model_dir = args.model_dir

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

log_dir = './log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    

      
        
def train(model, trainloader,metrics,criterion,optimizer,scheduler,G_support):

    train_loss_fft = 0.0

    
    model.train()
    
    for i, (true_amp, ft_images) in enumerate(trainloader):
        
        true_amp = true_amp.to(device)
        ft_images = ft_images.to(device) #Move everything to device
        G_support = G_support.to(device)
        


        output = model(ft_images) #Forward pass
        pred_re = output[..., 0]
        pred_im = output[..., 1]
        
        complex_x = torch.complex(pred_re,pred_im)
        
        pre_phase = torch.angle(complex_x)*G_support
        pre_amp = torch.abs(complex_x)*G_support
        
        pred_re_support = pre_amp*torch.cos(pre_phase)
        pred_im_support = pre_amp*torch.sin(pre_phase)
        
        complex_support= torch.complex(pred_re_support,pred_im_support)

        #Compute FT, shift and take abs
        y = torch.fft.fftshift(complex_support,dim=(-2,-1))
        y = torch.fft.fftn(y,dim=(-2,-1))
        y = torch.fft.ifftshift(y,dim=(-2,-1))
        y = torch.abs(y)
        

        #Compute losses
        loss_fft = criterion(y,true_amp)
        
        

        #Zero current grads and do backprop
        optimizer.zero_grad() 
        loss_fft.backward()
        optimizer.step()

      
        train_loss_fft += loss_fft.detach().item()


        #Update the LR according to the schedule 
        scheduler.step() 
        metrics['lrs'].append(scheduler.get_last_lr())
        
        
    #Divide cumulative loss by number of batches-- sli inaccurate because last batch is different size
    metrics['losses'].append([train_loss_fft/(i+1)])
    

def validate(model,validloader,metrics,epoch,criterion,optimizer,G_support):

    loss_fft = 0.0

    model.eval()
    
    for j, (true_amp,ft_images) in enumerate(validloader):
        true_amp = true_amp.to(device)
        ft_images = ft_images.to(device)
        G_support = G_support.to(device)
   


        
        with torch.no_grad():
            output = model(ft_images) #Forward pass
            pred_re = output[..., 0]
            pred_im = output[..., 1] 
            
            complex_x = torch.complex(pred_re,pred_im)
            
            pre_phase = torch.angle(complex_x)*G_support
            pre_amp = torch.abs(complex_x)*G_support
            
            pred_re_support = pre_amp*torch.cos(pre_phase)
            pred_im_support = pre_amp*torch.sin(pre_phase)
            
            complex_support= torch.complex(pred_re_support,pred_im_support)

            #Compute FT, shift and take abs
            y = torch.fft.fftshift(complex_support,dim=(-2,-1))
            y = torch.fft.fftn(y,dim=(-2,-1))
            y = torch.fft.ifftshift(y,dim=(-2,-1))
            y = torch.abs(y)
             
      
            val_loss_fft = criterion(y,true_amp)          
            loss_fft += val_loss_fft.detach().item()

        
    metrics['val_losses'].append([loss_fft/(j+1)])
  

all_images = np.load(args.real_img_path)
all_real = np.load(args.pretrain_real_path)
all_imagary = np.load(args.pretrain_imagry_path)

all_images = propcessing_img(all_images)

X_train_amp_tensor, X_train_tensor, X_test_amp_tensor, X_test_tensor = real_datasets(propcessing_img,all_real,all_imagary)

train_data = TensorDataset(X_train_amp_tensor, X_train_tensor)
test_data = TensorDataset(X_test_amp_tensor, X_test_tensor)

N_VALID = 0
N_TRAIN = X_train_tensor.shape[0]
train_data2, valid_data = torch.utils.data.random_split(train_data,[N_TRAIN-N_VALID,N_VALID])
print(len(train_data2),len(valid_data),len(test_data))



def main():
    model = Unet().to(device)
        
    BATCH_SIZE = 3
    LR = 0.001
    trainloader = DataLoader(train_data2, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    testloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    
    iterations_per_epoch = np.floor((N_TRAIN-N_VALID)/BATCH_SIZE)+1 #Final batch will be less than batch size
    step_size = 10*iterations_per_epoch #Paper recommends 2-10 number of iterations, step_size is half cycle
    print("LR step size is:", step_size, "which is every %d epochs" %(step_size/iterations_per_epoch))
    
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr/10, max_lr=args.lr, step_size_up=step_size,
                                                  cycle_momentum=False, mode='triangular2')
    
    metrics = {'losses':[],'val_losses':[], 'lrs':[], 'best_val_loss' : np.inf}
    sigx = 3
    sigy = 20
    G_support = generate_Gaussian_support(sigx,sigy)
    for epoch in range (args.epochs):
        
        #Training loop
        train(model, trainloader,metrics,criterion,optimizer,scheduler,G_support)
    
    
       
        print('Epoch: %d | FFT  | Train Loss: %.3f' %(epoch, metrics['losses'][-1][0]))
        print('Epoch: %d | Ending LR: %.6f ' %(epoch, get_lr(optimizer)))
            
    torch.save(model.state_dict(),os.path.join(model_dir, 'complex_AE+'.pth'))


if __name__ == '__main__':
    main()
    
