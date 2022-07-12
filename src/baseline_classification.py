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

parser = argparse.ArgumentParser(description='get baseline for classification task')
parser.add_argument('--stations_cam', help='list of stations with camera number, separated with underscore (e.g. Buelenberg_1')
parser.add_argument('--path_dset', help='path to used dataset ')
args = parser.parse_args()

# wandb.init(project="model_fog_classification", entity="jbaumer", config=args)


############ GLOBAL VARIABLES ############

PATH_DATASET = args.path_dset
STATIONS_CAM_STR = args.stations_cam

STATIONS_CAM_STR = STATIONS_CAM_STR.replace("\\", "")
STATIONS_CAM_LST = sorted(ast.literal_eval(STATIONS_CAM_STR))  # sort to make sure not two models with data from same cameras (but input in different order) will be saved


dset_test = DischmaSet_classification(root=PATH_DATASET, stat_cam_lst=STATIONS_CAM_LST, mode='test')
dset_baseline = DischmaSet_classification(root=PATH_DATASET, stat_cam_lst=STATIONS_CAM_LST, mode='baseline')

y_GT = dset_test.is_foggy
y_pred_baseline = dset_baseline.is_foggy

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
