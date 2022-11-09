# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 17:03:23 2022

@author: jehig

feature extraction nn branches then convolutional nn truck for rigid deformation parameters (trans1,trans2,rotate)
"""

import torch 
import torch.nn as nn
import math
from matplotlib import pyplot as plt

#single 5x5 convolution 
def conv_5(in_channel, out_channel):
    conv = nn.Sequential(
        nn.Conv2d(out_channel, out_channel, kernel_size=5, padding=2),
        nn.ReLU(inplace= True),
    )
    return conv

def conv_5_strd_2(in_channel, out_channel):
    conv = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=5, stride=2, padding=2),
        nn.ReLU(inplace= True),
    )
    return conv

#double 3x3 convolution 
def dual_conv(in_channel, out_channel):
    conv = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
        nn.ReLU(inplace= True),
        nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
        nn.ReLU(inplace= True),
    )
    return conv

#double 7x7 convolution 
def dual_conv_7(in_channel, out_channel):
    conv = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=7, padding=3),
        nn.ReLU(inplace= True),
        nn.Conv2d(out_channel, out_channel, kernel_size=7, padding=3),
        nn.ReLU(inplace= True),
    )
    return conv

#double 5x5 convolution 
def dual_conv_5(in_channel, out_channel):
    conv = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=5, padding=2),
        nn.ReLU(inplace= True),
        nn.Conv2d(out_channel, out_channel, kernel_size=5, padding=2),
        nn.ReLU(inplace= True),
    )
    return conv
'''
# crop the image(tensor) to equal size 
# as shown in architecture image , half left side image is concated with right side image
def crop_tensor(target_tensor, tensor):
    target_size1 = target_tensor.size()[2]
    tensor_size1 = tensor.size()[2]
    delta1 = tensor_size1 - target_size1
    delta1a = delta1 // 2
    delta1b = math.ceil(float(delta1) / 2)
    target_size2 = target_tensor.size()[3]
    tensor_size2 = tensor.size()[3]
    delta2 = tensor_size2 - target_size2
    delta2a = delta2 // 2
    delta2b = math.ceil(float(delta2) / 2)

    return tensor[:, :, delta1a:tensor_size1-delta1b, delta2a:tensor_size2-delta2b]
'''
class unet_3dwn_rid(nn.Module):
    def __init__(self):
        super(unet_3dwn_rid, self).__init__()

        # main trunk
        self.dwn_conv1 = dual_conv_7(2, 32)
        self.dwn_conv2 = dual_conv_5(32, 64)
        self.dwn_conv3 = dual_conv(64, 128)
        self.dwn_conv4 = dual_conv(128, 128)
        self.dwn_conv5 = dual_conv(128, 128)
        self.dwn_conv6 = dual_conv(128, 128)
        self.dwn_conv7 = dual_conv(128, 64)
        self.dwn_conv8 = dual_conv(64, 32)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        #output layer
        self.out = nn.Conv2d(32, 3, kernel_size=1)


    def forward(self, mov, fix):

        image = torch.concat((mov,fix), 1)        

        img192 = torch.nn.functional.interpolate(input=image, size=[192,192], scale_factor=None,  
                                                 mode='bilinear', align_corners=None, recompute_scale_factor=None)

        x1 = self.dwn_conv1(img192)
        x2 = self.maxpool(x1)
        x3 = self.dwn_conv2(x2)
        x4 = self.maxpool(x3)
        x5 = self.dwn_conv3(x4)
        x6 = self.maxpool(x5)
        x7 = self.dwn_conv4(x6)
        x8 = self.maxpool(x7)
        x9 = self.dwn_conv5(x8)
        x10 = self.maxpool(x9)
        x11 = self.dwn_conv6(x10)
        x12 = self.maxpool(x11)
        x13 = self.dwn_conv7(x12)
        x14 = self.maxpool(x13)
        x15 = self.dwn_conv8(x14)
    
        to = self.out(x15)
        
        return to
