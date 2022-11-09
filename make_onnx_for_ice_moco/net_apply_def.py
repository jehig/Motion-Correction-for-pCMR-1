# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 13:45:47 2022

@author: jehig
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

class net_apply_def(nn.Module):
    def __init__(self):
        super(net_apply_def, self).__init__()
        
    def warpimg_net(self, x, flo):
        
        flo = torch.flip(flo, dims=(1,)) 
        
        B, C, H, W = x.size()

        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid -1 to 1
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0 # vertical 
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0 # horizontal

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        
        return output

    def forward(self, imagea, def2a):
        
        imagea_mov = imagea[:,0,:,:] # first image in channel direction is moving
        imagea_mov = torch.unsqueeze(imagea_mov,1)
        
        imageTa = self.warpimg_net(imagea_mov, def2a)
        
        return imageTa
    
    
        