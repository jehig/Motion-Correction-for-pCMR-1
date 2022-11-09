# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 18:27:23 2022

@author: jehig

Code to create onnx file "pCMR_moco.onnx", containing a pipeline to correct perfusion cardiac
MRI data affected by respiratory motion. It is designed to tackle dramaticly changing contrast
over time. the pipeline consists of:
1) tree convolutional neural network for rigid registration parameters (trained using ~5000
   images with supervised learning)
2) pytorch implementation for turning rigid parameters to deformation field and applying
3) pytorch implementation for applying deformation field to moving image
4) U-net for calculating defomation field for deformable motion (trained using ~5000 images 
   with supervised learning)
5) pytorch implementation for applying deformation field to moving image

inference data format: format of tensor with moving and fixed images in channel direction
fixed image: the time series image with peak left ventrical contrast 
moving image: any other image in timeseries, with very different contrast regime 
"""

# generic import
import numpy as np 
from matplotlib import pyplot as plt

# torch import
import torch
import torch.nn as nn
import torch.nn.functional as F

# onnx import
import onnx
import onnxruntime

# custom functions
from rid_def_model import rid_def_model
from unet_3dwn_rid import unet_3dwn_rid
from make_rid_def import make_rid_def
from unet_3dwn import Unet
from net_apply_def import net_apply_def
from autocrop_img import autocrop_img


# load rigid transform model and trained state_dicts    
rid_model = unet_3dwn_rid()
chkpath = 'trained_nn_states/rigid_nn_epoch70.pth'
checkpoint = torch.load(chkpath,map_location='cpu')
rid_model.load_state_dict(checkpoint)

# load deformable registration model and trained state_dicts 
def_model = Unet()
chkpath = 'trained_nn_states/deform_nn_epoch50.pth'
checkpoint = torch.load(chkpath,map_location='cpu')
def_model.load_state_dict(checkpoint)

# load custom torch modules for applying rigid transforms and deformation fields 
make_rid_def_model = make_rid_def()
apply_def_model = net_apply_def()

# combine models into graph
model = rid_def_model(rid_model.float(), make_rid_def_model.float(), def_model.float(), apply_def_model.float())

# create onnx file for onnxruntime, with dynamic axes for variable dimensions
movinp = torch.zeros(1, 1, 15, 15)
fixinp = torch.zeros(1, 1, 15, 15)
dynamic_axes = {'input_1': [0, 1, 2, 3], 
                'input_2': [0, 1, 2, 3], 
                'output_1': [0, 1, 2, 3]}

torch.onnx.export(model,
                  (movinp,fixinp),
                  'pCMR_moco.onnx',
                  input_names=["input_1","input_2"], 
                  output_names=["output_1"],
                  dynamic_axes=dynamic_axes,
                  opset_version=16)

# load example data, in format of tensor with moving and fixed images in channel direction
# fixed image: the time series image with peak left ventrical contrast 
# moving image: any other image in timeseries, with very different contrast regime  
movfix = np.load('example_data/scan8_stage1_rep50_rnd1.npy')

# choose image dimensions to crop (networks work with variable image size)
newh = 126; neww = 126;
movfixc = np.random.randn(newh,neww,2)
movfixc[:,:,0] = autocrop_img(movfix[:,:,0],newh,neww)
movfixc[:,:,1] = autocrop_img(movfix[:,:,1],newh,neww)
#movfix = movfixc

# set up input for onnxruntime (numpy float32 in this case, may differ in scanner reconstruction environment) 
mf_np = np.transpose(movfix,(2,0,1))
mf_np = np.expand_dims(mf_np, 0)
mf_np = mf_np.astype(np.float32)
m_np = mf_np[:,0,:,:,]
m_np = np.expand_dims(m_np, 0)
f_np = mf_np[:,1,:,:,]
f_np = np.expand_dims(f_np, 0)

# run onnxruntime session to infer motion correction
ort_session = onnxruntime.InferenceSession("pCMR_moco.onnx")
ort_inputs = {ort_session.get_inputs()[0].name: m_np, ort_session.get_inputs()[1].name: f_np}
ort_outs = ort_session.run(None, ort_inputs)
output = ort_outs[0]

# show results: 
# 1) example moving image
# 2) reference image (peak left ventical contrast in time series) 
# 3) output moving image after registration to reference
plt.colorbar(plt.imshow(movfix[:,:,0]),cmap='gray')
plt.show()
plt.colorbar(plt.imshow(movfix[:,:,1]),cmap='gray')
plt.show()
plt.colorbar(plt.imshow(output[0,0,:,:]),cmap='gray')
plt.show()