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

from untils import _retrieve_elements_from_indices, get_lr
from model import Unet




parser = argparse.ArgumentParser(description='PyTorch Phase amplitude retrieval ')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train')

parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')


parser.add_argument('--model-dir', type=str, required=True)

parser.add_argument('--save-dir', default='./trained_model_test_AE',
                    help='directory of model for saving checkpoint')
                    
parser.add_argument('--data-dir', type=str, required=True)
                    
args = parser.parse_args()

device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')

model_dir = args.model_dir

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

log_dir = './log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    

        
def train(model, trainloader,metrics,criterion,optimizer,scheduler):
    tot_loss = 0.0
    train_loss_re = 0.0
    train_loss_im = 0.0

    
    model.train()
    
    for i, (ft_images,amps,phs) in enumerate(trainloader):
        ft_images = ft_images.to(device) #Move everything to device
        amps = amps.to(device)
        phs = phs.to(device)
        
        true_re = amps*torch.cos(phs)
        true_im = amps*torch.sin(phs)

        output = model(ft_images) #Forward pass
        pred_re = output[..., 0]
        pred_im = output[..., 1]

        #Compute losses
        loss_re =  criterion(pred_re,true_re) #Monitor amplitude loss
        loss_im = criterion(pred_im,true_im) #Monitor phase loss but only within support (which may not be same as true amp)
        
        
        
        loss = loss_re + loss_im  #Use equiweighted amps and phase

        #Zero current grads and do backprop
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()

        tot_loss += loss.detach().item()
        train_loss_re += loss_re.detach().item()
        train_loss_im += loss_im.detach().item()


        #Update the LR according to the schedule -- CyclicLR updates each batch
        scheduler.step() 
        metrics['lrs'].append(scheduler.get_last_lr())
        
        
    #Divide cumulative loss by number of batches-- sli inaccurate because last batch is different size
    metrics['losses'].append([tot_loss/i,train_loss_re/i,train_loss_im/i]) 
    

def validate(model,validloader,metrics,epoch,criterion,optimizer):
    tot_val_loss = 0.0
    loss_re = 0.0
    loss_im = 0.0

    model.eval()
    
    for j, (ft_images,amps,phs) in enumerate(validloader):
        ft_images = ft_images.to(device)
        amps = amps.to(device)
        phs = phs.to(device)
        true_re = amps*torch.cos(phs)
        true_im = amps*torch.sin(phs)


        
        with torch.no_grad():
            output = model(ft_images) #Forward pass
            pred_re = output[..., 0]
            pred_im = output[..., 1] 

            
        
            val_loss_re = criterion(pred_re,true_re)
            val_loss_im = criterion(pred_im,true_im)
            val_loss = val_loss_im + val_loss_re 
        
            tot_val_loss += val_loss.detach().item()
            
            loss_re += val_loss_re.detach().item() 
            loss_im += val_loss_im.detach().item()

        
    metrics['val_losses'].append([tot_val_loss/j,loss_re/j,loss_im/j])
  

#%%
object_phase = []
object_modulus = []
detector_modulus = []


synthetic_data_dir = args.data_dir


data_dir = synthetic_data_dir + "/*.npz"
print(data_dir)
for f in glob.glob(data_dir):
    data_file = np.load(f)
    object_phase.append(data_file['object_phase'])
    object_modulus.append(data_file['object_modulus'])
    detector_modulus.append(data_file['modulus_detector'])
    

obj_phase_array = np.array(object_phase,dtype=np.float32).reshape(-1,64,64)
object_modulus_array = np.array(object_modulus,dtype=np.float32).reshape(-1,64,64)
detector_modulus_array = np.array(detector_modulus,dtype=np.float32).reshape(-1,64,64)
object_modulus_array_sqrt = np.sqrt(detector_modulus_array) 

num_train = int(detector_modulus_array.shape[0]*0.8)
X_train = object_modulus_array_sqrt[:num_train,:,:][:,np.newaxis,:,:,np.newaxis]
X_train = X_train/np.max(X_train)
X_train_complex = np.concatenate((X_train, X_train), 4)

X_test = object_modulus_array_sqrt[num_train:,:,:][:,np.newaxis,:,:,np.newaxis]
X_test = X_test/np.max(X_test)
X_test_complex = np.concatenate((X_test, X_test), 4)


Y_I_train = object_modulus_array[:num_train,:,:][:,np.newaxis,:,:]
Y_I_test = object_modulus_array[num_train:,:,:][:,np.newaxis,:,:]


Y_phi_train = obj_phase_array[:num_train,:,:][:,np.newaxis,:,:]
Y_phi_test = obj_phase_array[num_train:,:,:][:,np.newaxis,:,:]

X_train, Y_I_train, Y_phi_train = shuffle(X_train_complex, Y_I_train, Y_phi_train, random_state=0)



#Training data
X_train_tensor = torch.Tensor(X_train) 
Y_I_train_tensor = torch.Tensor(Y_I_train) 
Y_phi_train_tensor = torch.Tensor(Y_phi_train)

#Test data
X_test_tensor = torch.Tensor(X_test) 
Y_I_test_tensor = torch.Tensor(Y_I_test) 
Y_phi_test_tensor = torch.Tensor(Y_phi_test)



train_data = TensorDataset(X_train_tensor,Y_I_train_tensor,Y_phi_train_tensor)
test_data = TensorDataset(X_test_tensor)

N_VALID = 1000
N_TRAIN = X_train_tensor.shape[0]
train_data2, valid_data = torch.utils.data.random_split(train_data,[N_TRAIN-N_VALID,N_VALID])
print(len(train_data2),len(valid_data),len(test_data))




def main():
    model = Unet().to(device)
    
    BATCH_SIZE = 64
    LR = 0.001
    trainloader = DataLoader(train_data2, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    validloader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    testloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    
    
    iterations_per_epoch = np.floor((N_TRAIN-N_VALID)/BATCH_SIZE)+1 #Final batch will be less than batch size
    step_size = 10*iterations_per_epoch #Paper recommends 2-10 number of iterations, step_size is half cycle
    print("LR step size is:", step_size, "which is every %d epochs" %(step_size/iterations_per_epoch))
    
    criterion = nn.L1Loss()


    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr)
    
    metrics = {'losses':[],'val_losses':[], 'lrs':[], 'best_val_loss' : np.inf}
    for epoch in range (args.epochs):
        
        #Training loop
        train(model, trainloader,metrics,criterion,optimizer,scheduler)
    
        #Validation loop
        validate(model,validloader,metrics,epoch,criterion,optimizer)
    
        print('Epoch: %d | Tot  | Train Loss: %.5f | Val Loss: %.5f' %(epoch, metrics['losses'][-1][0], metrics['val_losses'][-1][0]))
        print('Epoch: %d | Re | Train Loss: %.4f | Val Loss: %.4f' %(epoch, metrics['losses'][-1][1], metrics['val_losses'][-1][1]))
        print('Epoch: %d | Im  | Train Loss: %.3f | Val Loss: %.3f' %(epoch, metrics['losses'][-1][2], metrics['val_losses'][-1][2]))
        print('Epoch: %d | Ending LR: %.6f ' %(epoch, get_lr(optimizer)))
        
        torch.save(model.state_dict(),os.path.join(model_dir, 'Real_complex_AE_6_5.pth'))
        np.save('my_file_real_input.npy', metrics) 
            



if __name__ == '__main__':
    main()

