# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 10:48:43 2022

@author: xyu1
"""

import numpy as np
import math
import cmath
from matplotlib import pyplot as plt

def generate_ellipse(sigx,sigy,x1,y1):
    '''
    sigx: #X ellipse width in pixels (domain)
    sigy: #Y ellipse width
    x1: width of output
    y1: 
    '''
    rot=-(math.pi)*25/180    #rotation angle
    G1 = np.zeros([y1,x1])
    for i in range(y1):
        for j in range(x1):
            x=(i-x1/2)*math.sin(rot)+(j-x1/2)*math.cos(rot)
            y=(i-y1/2)*math.cos(rot)-(j-y1/2)*math.sin(rot)
            G1[i][j] = math.exp(-x**2/sigx**2-y**2/sigy**2) #Gaussian
            
    return G1

def generate_random_phase(seed,x1,y1,G1):
    '''
    Input:
    seed: value for the stochastic and reproducible.
    x1: width of domain
    y1: length of domain

    Output:
    A: Size of NxN continaing random generated phase.
    
    '''
    
    offset_scale = 10
    np.random.seed(seed)
    N = 1024
    A = np.zeros([N,N],dtype = 'complex')
    for i in np.arange(10,N-20,x1-3,dtype=int):
        for j in np.arange(10,N-20,x1-3,dtype=int):
            A[i:i+y1,j:j+x1] = A[i:i+y1,j:j+x1] + G1*cmath.exp(2j*cmath.pi*np.random.rand())
            #add offset 
            ox = int(offset_scale*(np.random.rand()-0.5))
            oy = int(offset_scale*(np.random.rand()-0.5))
            A[i+oy:i+oy+y1,j+ox:j+ox+x1] = A[i+oy:i+oy+y1,j+ox:j+ox+x1] + G1*cmath.exp(2j*math.pi*np.random.rand())
    return A


def generate_Gaussian_support(rot):
    '''
    Input:
    rot: the rotation angle of Gaussian

    Output:
    G2: the Gaussian support
    '''
    N2=64 #the output size N2xN2
    G2 = np.zeros([N2,N2])
    sigx=9   #X ellipse width in pixels (support)
    sigy=30    #Y ellipse width
    #rot=-math.pi*35/180     # rotation angle
    for i in range(N2):
        for j in range(N2):
            x=(i-N2/2)*math.sin(rot)+(j-N2/2)*math.cos(rot)
            y=(i-N2/2)*math.cos(rot)-(j-N2/2)*math.sin(rot)
            G2[i][j] = math.exp(-x**2/sigx**2-y**2/sigy**2) #Gaussian
    return G2

def Generate_synthetic_data(A,G2,N,N3):
    '''
    Generate synthetic data with random generated phase
    Input:
    A: random generated phase
    G2: Gaussian Support
    N: size of whole ramdom phase
    N3: phase interval

    Output:
    
    object_phase: phase in the real space
    object_modulus: amplitude in the real space
    modulus_detector: ampltude in the detector space


    '''
    
    N2 = G2.shape[0]
    object_phase = []
    object_modulus = []
    modulus_detector = []
    
    for i in np.arange(N/2-N3,N/2+N3,64,dtype=int):
        for j in np.arange(N/2-N3,N/2+N3,64,dtype=int):
            B = np.zeros([N,N],dtype = 'complex')
            B = A[i:i+N2,j:j+N2]*G2  #offset probe position
            
            object_phase.append(np.angle(B))
            object_modulus.append(abs(B))
            
            new_B = np.float32(abs(B))*np.exp(2j*math.pi*np.float32(np.angle(B)))
            C = np.fft.ifftshift(np.fft.fftn(np.fft.fftshift(new_B))) #fftn
            
            modulus_detector.append(abs(C))
            
    return object_phase, object_modulus, modulus_detector


sigx=1.5 #X domain ellipse width in pixels
sigy=3   #y domain ellipse width in pixels
x1 = 6   #width of domain
y1 = 8   #length  of domain
N = 1024 #size of NxN phase
N3 = 448 #phase interval


G1 = generate_ellipse(sigx,sigy,x1,y1)
G2 = generate_Gaussian_support(rot=-math.pi*35/180)

#%%Generate simulated data
for seed in range(100):
    file_name = "./synthetic_data/data"+str(seed) #file path for saving sythentic data
    A = generate_random_phase(seed,x1,y1,G1)
    #generated data
    object_phase, object_modulus, modulus_detector = Generate_synthetic_data(A,G2,N,N3)
    #save generated data into specific file name path
    np.savez(file_name,object_phase=object_phase,object_modulus = object_modulus,modulus_detector = modulus_detector)

