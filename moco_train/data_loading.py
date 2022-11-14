# -*- coding: utf-8 -*-
"""
Created on Mon Aug 8 12:15:10 2022

@author: jehig

A subclass of torch.utils.data.Dataset required for loading batches in loop
"""

import logging
from os import listdir
from os.path import splitext
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class BasicDataset(Dataset):
    # initiate instance of dataset class, required for loading batches 
    def __init__(self, images_dir: str, preds_dir: str, pred_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.preds_dir = Path(preds_dir)
        self.pred_suffix = pred_suffix
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No files found in {images_dir}')
        logging.info(f'Creating dataset of size {len(self.ids)}')

    # get dataset size 
    def __len__(self):
        return len(self.ids)

    # add file type options, although npy used for training data 
    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            np.load(filename).shape
            return np.load(filename)
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    # load images and ground truths from respective directories, where corresponding files have same name
    def __getitem__(self, idx):
        name = self.ids[idx]
        pred_file = list(self.preds_dir.glob(name + self.pred_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(pred_file) == 1, f'no pred or multiple preds with ID {name}: {pred_file}'
        assert len(img_file) == 1, f'no image or multiple images with ID {name}: {img_file}'
        pred = self.load(pred_file[0])
        img = self.load(img_file[0])

        assert img.size == pred.size, \
            'Image and pred {name} must be same size, but are sizes {img.size} and {pred.size}'

        img_ndarray = np.asarray(img)
        if img_ndarray.ndim == 2: 
            img = img_ndarray[np.newaxis, ...]
        pred_ndarray = np.asarray(pred)
        if pred_ndarray.ndim == 2: 
            pred = pred_ndarray[np.newaxis, ...]

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'target': torch.as_tensor(pred.copy()).float().contiguous() 
        }
