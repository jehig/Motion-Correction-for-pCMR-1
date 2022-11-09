# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 20:19:38 2022

@author: jehig

inputs: image as 2d array, intended new height, intende new width
output: image cropped if new dimensions smaller, 
        or border pixels reblicated if new dimensions bigger   
"""
import numpy as np
import cv2

def autocrop_img(img,newh,neww):
    
    old_height, old_width = img.shape
    
    if newh <= old_height:     
        
        crph = old_height - newh 
        padh1 = np.floor(crph/2)
        padh1 = padh1.astype(int)
        padh2 = np.ceil(crph/2)
        padh2 = old_height - padh2.astype(int)
        result1 = img[padh1:padh2,:]
    
    else:              
        padh = newh - old_height
        padh1 = np.floor(padh/2)
        padh1 = padh1.astype(int)
        padh2 = np.ceil(padh/2)
        padh2 = padh2.astype(int)
        result1 = cv2.copyMakeBorder(img, padh1, padh2, 0, 0, cv2.BORDER_REPLICATE)
        
    if neww <= old_width:     
        
        crpw = old_width - neww 
        padw1 = np.floor(crpw/2)
        padw1 = padw1.astype(int)
        padw2 = np.ceil(crpw/2)
        padw2 = old_width - padw2.astype(int)
        result2 = result1[:,padw1:padw2]
        
    else:       
        padw = neww - old_width
        padw1 = np.floor(padw/2)
        padw1 = padw1.astype(int)
        padw2 = np.ceil(padw/2)
        padw2 = padw2.astype(int)
        result2 = cv2.copyMakeBorder(result1, 0, 0, padw1, padw2, cv2.BORDER_REPLICATE)
    
    return result2