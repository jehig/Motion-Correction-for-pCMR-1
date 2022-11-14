# -*- coding: utf-8 -*-
"""
Created on Tue May 24 16:24:10 2022
@author: jehig
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
#import SimpleITK as sitk
#from matplotlib import pyplot as plt
#import numpy as np

def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)

    # loop over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        
        image, mask_true = batch['image'], batch['target']
        
        # images and ground truths to required type and device
        image = image.to(device=device, dtype=torch.float32)
        image = torch.permute(image, (0,3,1,2))
        mask_true = mask_true.to(device=device, dtype=torch.float32) 
        mask_true = torch.permute(mask_true, (0,3,1,2))

        with torch.no_grad():
            
            # predict the deformation fields
            mask_pred = net(image)
            
            # validation score is square of vector error in predicted 2d deformation fields against ground truth
            i_err = mask_pred[:,0,:,:] - mask_true[:,0,:,:]
            j_err = mask_pred[:,1,:,:] - mask_true[:,1,:,:]
            vec_err = torch.square(i_err) + torch.square(j_err)
            loss = torch.sum( vec_err )

    # Fixes potential divide by zero error
    if num_val_batches == 0:
        return loss
    return loss / num_val_batches
