# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 17:03:23 2022

@author: jehig

U-Net based Architecture for deformation field
"""

import torch 
import torch.nn as nn
import math
from matplotlib import pyplot as plt

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

# crop the image tensor to keep bridge equal size 
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

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        # Left side (contracting path)
        self.dwn_conv1 = dual_conv_7(2, 64)
        self.dwn_conv2 = dual_conv_5(64, 128)
        self.dwn_conv3 = dual_conv(128, 256)
        self.dwn_conv4 = dual_conv(256, 512)
        self.dwn_conv5 = dual_conv(512, 1024)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        #Right side  (expnsion path) 

        self.trans1 = nn.ConvTranspose2d(1024,512, kernel_size=2, stride= 2)
        self.up_conv1 = dual_conv(1024,512)

        self.trans2 = nn.ConvTranspose2d(512,256, kernel_size=2, stride= 2)
        self.up_conv2 = dual_conv(512,256)

        self.trans3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride= 2)
        self.up_conv3 = dual_conv_5(256,128)

        self.trans4 = nn.ConvTranspose2d(128,64, kernel_size=2, stride= 2)
        self.up_conv4 = dual_conv_7(128,64)

        #output layer
        self.out = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, mov, fix):

        # concatinate move and fix
        image = torch.concat((mov,fix), 1)            

        # resize image to training data resolution
        imgrs = torch.nn.functional.interpolate(input=image, size=[112,112], scale_factor=None,  
                                                 mode='bilinear', align_corners=None, recompute_scale_factor=None)

        # forward pass, left side
        x1 = self.dwn_conv1(imgrs)
        x2 = self.maxpool(x1)
        x3 = self.dwn_conv2(x2)
        x4 = self.maxpool(x3)
        x5 = self.dwn_conv3(x4)
        x6 = self.maxpool(x5)
        x7 = self.dwn_conv4(x6)
        x8 = self.maxpool(x7)
        x9 = self.dwn_conv5(x8)

        # forward pass, right side
        xa = self.trans1(x9)
        ya = crop_tensor(xa, x7)
        xb = self.up_conv1(torch.cat([xa,ya], 1))

        xc = self.trans2(xb)
        yb = crop_tensor(xc, x5)
        xd = self.up_conv2(torch.cat([xc,yb], 1))

        xe = self.trans3(xd)
        yc = crop_tensor(xe, x3)
        xf = self.up_conv3(torch.cat([xe,yc], 1))

        xg = self.trans4(xf)
        yd = crop_tensor(xg, x1)
        xh = self.up_conv4(torch.cat([xg,yd], 1))
        
        x = self.out(xh)
        
        # resize output to input resolution
        xout = torch.nn.functional.interpolate(input=x, size=[fix.shape[2],fix.shape[3]], scale_factor=None,  
                                               mode='bilinear', align_corners=None, recompute_scale_factor=None)
        
        return xout
