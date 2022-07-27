# jbaumer
# calculate metrics for different datasets (1x/cam + 1 for all cams)

import torch
import argparse
import os
import wandb
import ast
import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from dischma_set_classification import DischmaSet_classification
import torch
import torchvision
import argparse
import matplotlib.pyplot as plt
import os
import wandb
import ast
import random
import math

from time import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, precision_recall_curve


from torch.utils.data import DataLoader, random_split
from torch import nn
from torchvision import models
from torch.optim import lr_scheduler

from dischma_set_segmentation import DischmaSet_segmentation

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

import warnings
import rasterio
from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('imports done')


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

parser = argparse.ArgumentParser(description='get baseline for segmentation task')
parser.add_argument('--stations_cam', help='list of stations with camera number, separated with underscore (e.g. Buelenberg_1')
parser.add_argument('--path_dset', help='path to used dataset ')
args = parser.parse_args()

# wandb.init(project="model_fog_segmentation", entity="jbaumer", config=args)


############ GLOBAL VARIABLES ############

PATH_DATASET = args.path_dset
STATIONS_CAM_STR = args.stations_cam

STATIONS_CAM_STR = STATIONS_CAM_STR.replace("\\", "")
STATIONS_CAM_LST = sorted(ast.literal_eval(STATIONS_CAM_STR))  # sort to make sure not two models with data from same cameras (but input in different order) will be saved


dset_test = DischmaSet_segmentation(root=PATH_DATASET, stat_cam_lst=STATIONS_CAM_LST, mode='test')
dset_baseline = DischmaSet_segmentation(root=PATH_DATASET, stat_cam_lst=STATIONS_CAM_LST, mode='baseline')

assert(len(dset_test) == len(dset_baseline))

dloader_test = DataLoader(dataset=dset_test, batch_size=1)  # 1 to avoid memory issues
dloader_baseline = DataLoader(dataset=dset_baseline, batch_size=1)

y_true_stack_total = torch.Tensor().to('cuda')
y_baseline_stack_total = torch.Tensor().to('cuda')

y_true_flat_all = {}

for idx, (x, y) in enumerate(dloader_test):
    y_true_flat = y.flatten()  # contains 0 (nodata), 1 (snow), 2 (nosnow)

    y_true_flat_all[idx] = y_true_flat 

    y_true_data = y_true_flat[y_true_flat_all[idx] != 0]  # contains 1, 2
    y_true_data[y_true_data==1] = 0  # convert ones to zeros
    y_true_data[y_true_data==2] = 1  # convert twos to ones
    # y_true_data now contains 0 (snow), 1 (no_snow)

    y_true_stack_total = torch.cat((y_true_stack_total, y_true_data), 0)  # append to flattened torch tensor 

for idx, (x, y) in enumerate(dloader_baseline):
    y_baseline_flat = y.flatten()  # contains 0 (nodata), 1 (snow), 2 (nosnow)

    y_baseline_data = y_baseline_flat[y_true_flat_all[idx] != 0]  # contains 1, 2
    y_baseline_data[y_baseline_data==1] = 0  # convert ones to zeros
    y_baseline_data[y_baseline_data==2] = 1  # convert twos to ones
    # y_baseline_data now contains 0 (snow), 1 (no_snow)

    y_baseline_stack_total = torch.cat((y_baseline_stack_total, y_baseline_data), 0)  # append to flattened torch tensor 

# y_true_stack_total contains GT
# y_baseline_stack_total contains baseline predictions

y_GT = y_true_stack_total.cpu()
y_pred_baseline = y_baseline_stack_total.cpu()

print('length of y GT: ', len(y_GT), 'length of y pred baseline: ', len(y_pred_baseline))
acc = accuracy_score(y_GT, y_pred_baseline)
prec = precision_score(y_GT, y_pred_baseline)
rec = recall_score(y_GT, y_pred_baseline)
f1 = f1_score(y_GT, y_pred_baseline)

print('baseline classificaton metrics from cam(s): ', STATIONS_CAM_LST)
print('accuracy: ', acc)
print('precision: ', prec)
print('recall: ', rec)
print('f1 score: ', f1)
