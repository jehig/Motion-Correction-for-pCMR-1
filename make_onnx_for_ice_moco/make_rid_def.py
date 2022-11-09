# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 13:45:47 2022

@author: jehig
"""

import torch 
import torch.nn as nn
import math
from matplotlib import pyplot as plt
import torchvision.transforms.functional as TF

class make_rid_def(nn.Module):
    def __init__(self):
        super(make_rid_def, self).__init__()

    def forward(self, dumimg, rid3):
        
        # deg to radians
        deg = rid3[0,2,0,0]
        rad = deg*2*3.14159/360
    
        # dimensions of dummy input
        B, C, H, W = dumimg.size()
        
        # center of mass
        comh = torch.ceil(H/2)
        comw = torch.ceil(W/2)
        
        # make mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1) - comw
        yy = yy.view(1,1,H,W).repeat(B,1,1,1) - comh
        grid = torch.cat((yy,xx),1).float()
        
        # create flow field and add rotation 
        riddef = torch.zeros((1,2,H,W))
        riddef[:,0,:,:] = grid[:,0,:,:]*torch.cos(rad) - grid[:,1,:,:]*torch.sin(rad) - grid[:,0,:,:]
        riddef[:,1,:,:] = grid[:,0,:,:]*torch.sin(rad) + grid[:,1,:,:]*torch.cos(rad) - grid[:,1,:,:]
        
        # add translation to flow field 
        riddef[:,0,:,:] = riddef[:,0,:,:] + rid3[0,0,0,0]
        riddef[:,1,:,:] = riddef[:,1,:,:] + rid3[0,1,0,0]
        
        return riddef


