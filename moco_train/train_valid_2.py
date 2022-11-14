# -*- coding: utf-8 -*-
"""
Created on Mon May 23 09:25:10 2022

@author: jehig

Trains a convolutional neural network model to predict deformation fields to register perfusion 
CMR dynamic images to the image in the time series with peak left vertical contrast.
The augmented sample training data represents simulated respiratory motion to correct, which is
only 0.3% of the ideal dataset.
"""

# Dependencies:
# Install CUDA
# Install PyTorch
# Install dependencies
# pip install -r requirements.txt

import numpy as np 
import argparse
import logging
import sys
import glob
from pathlib import Path
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import math
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from matplotlib import pyplot as plt

from data_loading import BasicDataset
from evaluate import evaluate
from unet_3dwn import Unet

# image and target deformation dirs should have equal number of identically named files
dir_img = Path('./sample_data/movfix')
dir_def = Path('./sample_data/def2d')

# set model dump directory
dir_checkpoint = Path('./checkpoints/training_run_1/')
# directory where partially trained models are loaded from if desired
dir_checkpoint_srt = dir_checkpoint

def train_net(net,
              device,
              chkload: int = 0,
              data_percent: int = 1,
              epochs: int = 1,
              batch_size: int = 100,
              learning_rate: float = 0.001,
              update_lr_sch: bool = True,
              use_lr_sch:bool = True,
              noise: float = 0,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              eval_rate: int = 10,
              amp: bool = False,
              alpha: float = 0):
    
    # 1. Create dataset
    dataset = BasicDataset(dir_img, dir_def) # no def_suffix
    # reduce dataset size for testing
    n_keep = int(len(dataset) * data_percent)
    n_remove = len(dataset) - n_keep
    dataset, removed_set = random_split(dataset, [n_keep, n_remove], generator=torch.Generator().manual_seed(0))

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    # fix generator with manual seed for reproducible result
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # Initialize logging
    experiment = wandb.init()
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    
    # 5. Set up learning rate scheduler, reduce learning rate if loss stops decreasing according to this rule
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.75)  

    # 6. load previous model if desired
    # if chkload = 0 start new model
    # if >0 choose saved checkpoint and optimiser parameters from directory 'dir_checkpoint_srt' with corresponding epoch number
    if chkload > 0:
        chkpathl = glob.glob( ''.join([os.getcwd(), '/', str(dir_checkpoint_srt), '/checkpoint_epoch', str(chkload),'_*.pth']) )
        chkpath = chkpathl[0]
        checkpoint = torch.load(chkpath)
        net.load_state_dict(checkpoint)
        
        optpathl = glob.glob( ''.join([os.getcwd(), '/', str(dir_checkpoint_srt), '/optimizer_epoch', str(chkload),'_*.pth']) )
        optpath = optpathl[0]
        checkpoint = torch.load(optpath)
        optimizer.load_state_dict(checkpoint)

        # true if learning rate from loaded model file should be used instead of intial value
        if use_lr_sch == False:
            optimizer.param_groups[0]['lr'] = learning_rate

    # 7. Begin training
    # number of epoch loops, with starting epoch greater than 0 if checkpoint was loaded
    epoch_range = range(chkload,epochs)
    # epoch loop
    for epoch in epoch_range:
        
        if epoch != chkload:     
            net.train()
        
        # for progress bar generation
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                               
                # load moving and fixed image pairs
                images = batch['image']
                # load groud truth 2d deformation fields
                true_defs = batch['target']

                # add noise for extra augmentation, if desired
                images = images + noise*torch.randn(images.shape)
                
                # mount images to device
                images = images.to(device=device, dtype=torch.float32)
                # permute so moving and fixed image pairs are joined in channel dimension
                images = torch.permute(images, (0,3,1,2))
                
                # mount images to device
                true_defs = true_defs.to(device=device, dtype=torch.float32)
                # mount ground truth to device
                true_defs = torch.permute(true_defs, (0,3,1,2))

                # sets the gradients of all optimized tensors to 0.
                optimizer.zero_grad()
                
                # with torch.cuda.amp.autocast(enabled=amp):
                defs_pred = net(images)                
                
                # calculate vector squared error of predicted deformation fields verses ground truth
                i_err = defs_pred[:,0,:,:] - true_defs[:,0,:,:]
                j_err = defs_pred[:,1,:,:] - true_defs[:,1,:,:]
                vec_err2 = ( torch.square(i_err) + torch.square(j_err) )
                
                # add smoothness term to cost function if desired, by having 'alpha' > 0
                # else skip this step before calculating total loss divided by batch size
                #alpha = 0.00
                if alpha > 0:
                    # smoothness crterion = magnitude of jacobian 
                    i_gradtup = torch.gradient(defs_pred[:,0,:,:])
                    j_gradtup = torch.gradient(defs_pred[:,1,:,:])
                    smth = torch.sum( torch.square(i_gradtup[1]) ) + torch.sum( torch.square(i_gradtup[2]) ) \
                           + torch.sum( torch.square(j_gradtup[1]) ) + torch.sum( torch.square(j_gradtup[2]) )       
                    print(alpha*smth)
                
                    loss = ( torch.sum( vec_err2 ) + alpha*smth ) / batch_size
                else:
                    loss = torch.sum( vec_err2 ) / batch_size
                
                # print loss
                print(loss)
                
                # compute gradient of loss in weights and bias space
                loss.backward()
                
                # update optimisation parameters
                optimizer.step()
                
                # update progress bar
                pbar.update(images.shape[0])
                pbar.set_postfix(**{'batch loss': loss.item()})

                
        # Evaluation round
        # 'eval_rate' rate in epochs at which evaluation and checkpoint saving occurs
        if (epoch+1)%eval_rate == 0:

            # validation score calculation, increases over epochs indicates overfitting
            val_score = evaluate(net, val_loader, device)
            if update_lr_sch == True:
                scheduler.step(val_score)

            # update log on evaluation
            logging.info('Eval score: {}'.format(val_score))
            experiment.log({
                'learning rate': optimizer.param_groups[0]['lr'],
                'validation Err': val_score,
                'images': wandb.Image(images[0].cpu()),
                'defs': {
                    'true': wandb.Image(true_defs[0].float().cpu()),
                    'pred': wandb.Image(torch.softmax(defs_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                },
                'epoch': epoch,
            })
            
            # normalise validation score by batch size and mutiply by 10000 to be readable as a text integer
            val_score_out = 10000 * val_score / batch_size
            # output learning rate at current epoch
            lr1 = (optimizer.param_groups[0]['lr'])
        
            # take dump, which includes val score and lr in name of saved checkpoint and optimisation file
            if save_checkpoint:
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{ep}_val{va}.pth'.format(ep = epoch + 1,va = int(val_score_out))))
                torch.save(optimizer.state_dict(), str(dir_checkpoint / 'optimizer_epoch{ep}_val{va}_al{al}_lr{le}.pth'
                                                       .format(ep = epoch + 1,
                                                               va = int(val_score_out),
                                                               al=alpha,
                                                               le = round(lr1, 3 - int(math.floor(math.log10(abs(lr1)))) - 1))))
                logging.info(f'Checkpoint {epoch + 1} saved!')

if __name__ == '__main__':

    # device is first GPU managed by cuda if available, otherwise use CPU 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load network architecture from file 
    net = Unet()
    net.to(device=device)

    train_net(net=net,
              data_percent=1,
              epochs=100000,
              batch_size=3,
              learning_rate=5e-06,
              update_lr_sch=True,
              noise = 0.01,
              device=device,
              val_percent=0.1,
              amp=False,
              alpha=0,
              eval_rate = 1)
