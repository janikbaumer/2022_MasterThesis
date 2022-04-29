import torch
import torchvision
import argparse
import matplotlib.pyplot as plt
import os
import wandb
import ast
import random

from time import time
import numpy as np
from numpy import datetime_as_string
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score


from torch.utils.data import DataLoader, random_split
from torch import nn
from torchvision import models
from torch.optim import lr_scheduler

from dischma_set_segmentation import DischmaSet_segmentation

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

import cv2

def save_x_y(x, y):
    """
    x.shape: torch.Size([batchsize, 3, 4000, 5984])
    y.shape: torch.Size([batchsize, 1, 4000, 6000])
    # note: x image will get very large

    """
    plt.imsave(fname='../test_x_03.png', arr=np.transpose(x[0].numpy(), (1,2,0)))
    y[y == 1] == 255  # snow -> white
    y[y == 3] == 125  # no data -> gray (no snow will stay 0 -> black)
    cv2.imwrite('../test_y_03.png', np.transpose(y[0].numpy(), (1,2,0)))
    # to show y[0], so first element of label batch: 
    # plt.imshow(np.transpose(y[0].numpy(), (1,2,0)))

def train_val_model(model, criterion, optimizer, scheduler, num_epochs):
    for epoch in range(num_epochs):

        for phase in ['train', 'val']:  # in each epoch, do training and validation
            print(f'{phase} phase in epoch {epoch+1}/{num_epochs} starting...')

            for x, y in dloader_train:
                save_x_y(x, y)

                x = x.to(device)
                y = y.to(device)

                x = x.float()
                y = y.float()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    y_pred = model(x)  # torch.Size([8, 3, 256, 256])
                    y_pred = y_pred.argmax(axis=1) # for each batch image, 
                    y_pred[y_pred == 2] = 3  # because no_data = 3 (two was only the index)
                    

                    # x_arr = x.flatten()  # TODO: x was 3D, y is only 1D !!!
                    y_true_list = list(y.flatten().cpu().numpy())
                    y_pred_list = list(y_pred.flatten().cpu().numpy())
                    loss = criterion(..., ...)  # TODO
                    

                    loss = criterion()
                    # show predicted image
                    # plt.imshow(np.transpose(y_pred[0].cpu().numpy().reshape(1, *y_pred.shape[1:]), (1,2,0)))

                    if phase == 'train':
                        loss.backward()  # backprop
                        optimizer.step()  # update params

                print()


# try U-Net for image segmentation

############ REPRODUCIBILITY ############

# Fix all random seeds
random_seed = 42
torch.seed()  # only for CPU
torch.manual_seed(random_seed)  # should work for CPU and CUDA
random.seed(random_seed)
np.random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(random_seed)

############ ARGPARSERS ############

parser = argparse.ArgumentParser(description='Run pretrained finetuning.')
parser.add_argument('--batch_size', type=int, help='batch size')
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--epochs', type=int, help='number of training epochs')
parser.add_argument('--train_split', type=float, help='train split')
parser.add_argument('--stations_cam', help='list of stations with camera number, separated with underscore (e.g. Buelenberg_1')
parser.add_argument('--weighted', help='how to weight the classes (manual: as given in script / Auto: Inversely proportional to occurance / False: not at all')
parser.add_argument('--path_dset', help='path to used dataset ')
parser.add_argument('--lr_scheduler', help='whether to use a lr scheduler, and if so after how many epochs to reduced LR')

args = parser.parse_args()

LOGGING = False
# logging
if LOGGING:
    wandb.init(project="snow_segmentation", entity="jbaumer", config=args)

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


############ GLOBAL VARIABLES ############

BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
EPOCHS = args.epochs
TRAIN_SPLIT = args.train_split
# WEIGHTED = args.weighted
PATH_DATASET = args.path_dset
LR_SCHEDULER = args.lr_scheduler

STATIONS_CAM_STR = args.stations_cam
STATIONS_CAM_STR = STATIONS_CAM_STR.replace("\\", "")
STATIONS_CAM_LST = sorted(ast.literal_eval(STATIONS_CAM_STR))  # sort to make sure not two models with data from same cameras (but input in different order) will be saved

#N_CLASSES = 2
PATH_MODEL = f'models/segmentation/{STATIONS_CAM_LST}_bs_{BATCH_SIZE}_LR_{LEARNING_RATE}_epochs_{EPOCHS}_lr_sched_{LR_SCHEDULER}'

print()

# create datasets and dataloaders
dset_train = DischmaSet_segmentation(root=PATH_DATASET, stat_cam_lst=STATIONS_CAM_LST, mode='train')
dset_val = DischmaSet_segmentation(root=PATH_DATASET, stat_cam_lst=STATIONS_CAM_LST, mode='val')
print(f'Dischma sets (train and val) with data from {STATIONS_CAM_LST} created.')

len_dset_train, len_dset_val = len(dset_train), len(dset_val)

dloader_train = DataLoader(dataset=dset_train, batch_size=BATCH_SIZE)
dloader_val = DataLoader(dataset=dset_val, batch_size=BATCH_SIZE)

# class 0: is not foggy
# class 1: is foggy
# class 3: no data

# TODO get balancedness of dset

# TODO ev consider weighting of classes


# model
model = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2,                      # model output channels (number of classes in your dataset)
)
model = model.to(device)

"""
preprocess_input = get_preprocessing_fn('resnet34', pretrained='imagenet')
x_try = torch.randn(size=(8, 3, 512, 512), dtype=torch.float32)
x_try = preprocess_input(x_try)
out = model(x_try)
"""

# loss functions to try: BCE / IoU-loss / focal loss
criterion = nn.CrossEntropyLoss(reduction='mean')  # TODO: currently, all occurances are considered, optimal would be to only consider occ. of train split
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)  # TODO ev add momentum

exp_lr_scheduler = None

print()

train_val_model(model=model, criterion=criterion, optimizer=optimizer, scheduler=exp_lr_scheduler, num_epochs=EPOCHS)

