# data loader
from __future__ import print_function, division
import glob
import torch
from skimage import io, transform, color
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import cv2
# ==========================dataset load==========================


class RescaleT(object):

    def __init__(self, output_size = 540):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        imidx, image = sample['imidx'], sample['image']

        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size*h/w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size*w/h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(
            image, (new_h, new_w), mode='constant')

        return {'imidx': imidx, 'image': img}

        return {'imidx': imidx, 'image': img, 'label': lbl}


class ToTensorLab(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, flag=0):
        self.flag = flag

    def __call__(self, sample):

        imidx, image = sample['imidx'], sample['image']
        tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
        if image.shape[2] == 1:
            tmpImg[:, :, 0] = (image[:, :, 0]-0.485)/0.229
            tmpImg[:, :, 1] = (image[:, :, 0]-0.485)/0.229
            tmpImg[:, :, 2] = (image[:, :, 0]-0.485)/0.229
        else:
            tmpImg[:, :, 0] = (image[:, :, 0]-0.485)/0.229
            tmpImg[:, :, 1] = (image[:, :, 1]-0.456)/0.224
            tmpImg[:, :, 2] = (image[:, :, 2]-0.406)/0.225
        tmpImg = tmpImg.transpose((2, 0, 1))

        return {'imidx': torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg)}

class SalObjDataset(Dataset):
    def __init__(self,img_name_list, transform=None):
        self.image_name_list = img_name_list
        self.transform = transform

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self,idx):
        #print(self.image_name_list[idx])
        image = io.imread(self.image_name_list[idx])
        
        imname = self.image_name_list[idx]
        imidx = np.array([idx])

        if 2==len(image.shape):
            image = image[:,:,np.newaxis]
            
        sample = {'imidx':imidx, 'image':image}

        if self.transform:
            sample = self.transform(sample)

        return sample
