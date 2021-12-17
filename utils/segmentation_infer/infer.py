import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob
import argparse
import random
import cv2
import tqdm
import shutil
import os

from .data_loader import RescaleT, ToTensorLab, SalObjDataset
from .model import U2NET
from .post_process import mask_refine


def get_file_list(root_dir):
    tra_img_name_list = []
    filenames = os.listdir(root_dir)
    filenames.sort()
    for filename in filenames:
        if ".png" in filename or ".jpg" in filename or ".JPG" in filename:
            tra_img_name_list.append(os.path.join(root_dir, filename))
    return tra_img_name_list


def get_lasted_model_file(local_dir):
    local_file = ""
    for file_name in os.listdir(local_dir):
        if file_name.endswith("pth"):
            if file_name > local_file:
                local_file = file_name
    return local_file



def segmentation_infer(data_dir="", model_dir="", rescale_size=540):

    rgb_dir = os.path.join(data_dir, "images_ori")
    #mask_dir = os.path.join(args.data_dir, "mask")
    white_bg_dir = os.path.join(data_dir, "images")


    test_img_name_list = get_file_list(rgb_dir)

    if len(test_img_name_list) == 0:
        exit(0)

    #if not os.path.exists(mask_dir):
    #    os.mkdir(mask_dir)
    if not os.path.exists(white_bg_dir):
        os.mkdir(white_bg_dir)

    test_salobj_dataset = SalObjDataset(
        img_name_list=test_img_name_list,
        transform=transforms.Compose([RescaleT(rescale_size),
                                      ToTensorLab(flag=0)]))

    test_salobj_dataloader = DataLoader(
        test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1)

    net = U2NET(3, 1)

    local_file = get_lasted_model_file(model_dir)
    if local_file == "":
        print("Model file not exits.")
        exit(-1)
    restore_from = model_dir + local_file
    if restore_from != "":
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(restore_from))
        else:
            net.load_state_dict(torch.load(
                restore_from, map_location=torch.device('cpu')))

    if torch.cuda.is_available():
        net.cuda()

    net.eval()
    # Eval
    for i_test, data_test in enumerate(tqdm.tqdm(test_salobj_dataloader)):
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)
        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        src_rgb = cv2.imread(test_img_name_list[i_test], 1)
        src_h, src_w, _ = src_rgb.shape


        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        file_name = test_img_name_list[i_test].split('/')[-1]

        if torch.cuda.is_available():
            pred_mask = np.squeeze(d1[0].cpu().detach().numpy())
        else:
            pred_mask = np.squeeze(d1[0].detach().numpy())

        pred_mask = pred_mask * 255
        

        pred_mask = cv2.resize(pred_mask, (src_w, src_h),
                               interpolation=cv2.cv2.INTER_LINEAR)
        pred_mask = np.uint8(pred_mask)

        pred_mask = mask_refine(pred_mask)

        pred_mask_save = np.copy(pred_mask)
        pred_mask_save[pred_mask_save >= 128] = 255
        pred_mask_save[pred_mask_save < 128] = 0

#        cv2.imwrite(os.path.join(mask_dir,
#                                 file_name), pred_mask_save)

        src_rgb[pred_mask < 128] = 255
        cv2.imwrite(os.path.join(white_bg_dir,
                                  file_name), src_rgb)

