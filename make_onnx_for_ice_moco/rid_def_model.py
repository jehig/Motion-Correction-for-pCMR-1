# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 16:11:45 2022

@author: jehig
"""

import torch 
import torch.nn as nn

from matplotlib import pyplot as plt

class rid_def_model(nn.Module):
    
    def __init__(self, rid_model, make_rid_def, def_model, apply_def):
        super(rid_def_model, self).__init__()
        self.rid_model = rid_model
        self.make_rid_def = make_rid_def
        self.def_model = def_model
        self.apply_def = apply_def
        
    def forward(self, mov, fix):
        rid = self.rid_model(mov,fix)
        rid[:,0,:,:] = rid[:,2,:,:]*30 # ground truth value divided by 30 during training 
        rid[:,1,:,:] = rid[:,2,:,:]*30 # ground truth value divided by 30 during training 
        rid[:,2,:,:] = rid[:,2,:,:]*360 # ground truth value was a fraction of 360 
        ridflo = self.make_rid_def(mov,rid)
        movrid = self.apply_def(mov,ridflo)
        
        defflo = self.def_model(movrid, fix)
        defflo = defflo*30 # ground truth deformation field divided by 30 during training 
        movflo = self.apply_def(movrid, defflo)
        return movflo