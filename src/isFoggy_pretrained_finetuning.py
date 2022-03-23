import sklearn
import torch
import torchvision
import argparse
import matplotlib.pyplot as plt
import json
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

from binary_classifier import MyNet
from dischma_set import DischmaSet

print('imports done')

################# FUNCTIONS ######################

def get_and_print_stats(confmat, mode=None, label_isFoggy=None):
    if mode == 'train':
        print('TRAINING METRICS: ')
    elif mode == 'val':
        print('VALIDATION METRICS: ')

    tn, fp, fn, tp = confmat.ravel()

    acc = (tn+tp)/(tn+fp+tp+fn)  # acc represents the number of correctly classified data instances over the total number of data instances

    if label_isFoggy == 1:  # if isFoggy is considered as the True label
        prec_isFoggyIsTrue = tp/(tp+fp)  # goal to keep fp low (to get high precision)
        print('prec_isFoggyIsTrue: \n', prec_isFoggyIsTrue, '\n')

        rec_isFoggyIsTrue = tp/(tp+fn)  # goal to keep fn low (to get high recall)
        print('rec_isFoggyIsTrue: \n', rec_isFoggyIsTrue, '\n')

        f1_isFoggyIsTrue = 2*(prec_isFoggyIsTrue*rec_isFoggyIsTrue)/(prec_isFoggyIsTrue+rec_isFoggyIsTrue)
        print('f1_isFoggyIsTrue: \n', f1_isFoggyIsTrue, '\n')

        return acc, prec_isFoggyIsTrue, rec_isFoggyIsTrue, f1_isFoggyIsTrue

    if label_isFoggy == 0:  # if isFoggy is considered as the False label
        prec_isFoggyIsFalse = tn/(tn+fn)  # goal to keep fp low (to get high precision)
        print('prec_isFoggyIsFalse: \n', prec_isFoggyIsFalse, '\n')

        rec_isFoggyIsFalse = tn/(tn+fp)  # goal to keep fn low (to get high recall)
        print('rec_isFoggyIsFalse: \n', rec_isFoggyIsFalse, '\n')

        f1_isFoggyIsFalse =  2*(prec_isFoggyIsFalse*rec_isFoggyIsFalse)/(prec_isFoggyIsFalse+rec_isFoggyIsFalse)
        print('f1_isFoggyIsFalse: \n', f1_isFoggyIsFalse, '\n')

        return acc, prec_isFoggyIsFalse, rec_isFoggyIsFalse, f1_isFoggyIsFalse

def get_train_val_split(dset_full):
    print('splitting in train/test...')
    len_full = len(dset_full)
    len_train = int(TRAIN_SPLIT*len_full)
    len_val = len_full - len_train
    dset_train, dset_val = random_split(dset_full, [len_train, len_val])  # Split Pytorch tensor
    return dset_train, dset_val

def print_grid(x,y, batchsize, loop):
    x = x.cpu()
    y = y.cpu()
    # print(y)
    y_reshaped = y.reshape(2, -1).numpy()
    grid_img = torchvision.utils.make_grid(x, nrow=int(batchsize/2), normalize=True)
    plt.title(f'loop: {loop}\n{y_reshaped[0]}\n{y_reshaped[1]}')
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.savefig(f'stats/fig_check_manually/grid_loop_{loop}')

def train_val_model(model, criterion, optimizer, scheduler, num_epochs):
    time_start = time()

    for epoch in range(num_epochs):
        print()
        print('-' * 10)

        # in each epoch, do training and validation
        for phase in ['train', 'val']:

            print(f'{phase} phase in epoch {epoch+1}/{num_epochs} starting...')
            if phase == 'train':
                model.train()
                dloader = dloader_train
            else:
                model.eval()
                dloader = dloader_val

            running_loss = 0.0  # loss (to be updated during loop)
            loop, loop_loss, epoch_loss = 0, 0, 0
            cm_tot = np.zeros((2,2), dtype='int64')

            for x, y in dloader:
                
                # move to GPU
                x = x.to(device)
                y = y.to(device)

                # plot some batches
                #if loop < 200 and loop%10 == 0:
                #    print_grid(x,y, BATCH_SIZE, loop)

                # zero the parameter gradients (not keep them from last batch)
                optimizer.zero_grad()

                # forward pass, track history (calc gradients) only in training
                with torch.set_grad_enabled(phase == 'train'):
                    pred = model(x)  # shape: batchsize, nclasses
                    pred_binary = pred.argmax(dim=1)  # to compare to y (y_true)
                    loss = criterion(pred, y)

                    if phase == 'train':
                        loss.backward()  # backprop
                        optimizer.step()  # update params

                # statistics
                running_loss += loss.item() * x.size(0)  # loss*batchsize (as loss was averaged ('mean')), each item of batch had this loss on avg
                loop_loss += loss.item() * x.size(0)

                y_true = y.cpu().tolist()
                y_pred = pred_binary.cpu().tolist()

                # array([[a, b],
                #        [c, d]])
                # a: tn (0 true, 0 predicted)  # negative is predicted, and the prediction is true
                # b: fp (0 true, 1 predicted)  # positive is predicted, and the prediction is wrong
                # c: fn (1 true, 0 predicted)  # negative is predicted, and the prediction is wrong
                # d: tp (1 true, 1 predicted)  # positive is predicted, and the prediction is true
                # to extract all values:
                # (tn, fp, fn, tp) = cm.ravel()
                cm_current = confusion_matrix(y_true, y_pred)
                cm_tot = cm_tot + cm_current  # accumulate confusion matrices

                loop += 1
                if (loop%200) == 0:
                    print(f'loop: {loop} / {len(dloader)} ... {phase} loss (avg over these 200 loops): ', loop_loss/200)
                    loop_loss = 0


            if phase == 'train':
                epoch_loss = running_loss/len_dset_train
                if scheduler is not None:
                    scheduler.step()
                    # scheduler.print_lr()

            elif phase == 'val':
                epoch_loss = running_loss/len_dset_val

            print('epoch loss = ', epoch_loss)

            acc, prec_isFoggyIsTrue, rec_isFoggyIsTrue, f1_isFoggyIsTrue = get_and_print_stats(confmat=cm_tot, mode=phase, label_isFoggy=1)
            
            # TODO: do every 200th batch iteration or so
            wandb.log({
                f'{phase} loss' : epoch_loss,
                f'{phase} accuracy' : acc,
                f'{phase} precision (isFoggy is True)' : prec_isFoggyIsTrue,
                f'{phase} recall (isFoggy is True)' : rec_isFoggyIsTrue,  # this should be high !!! (to catch all (foggy) images)
                f'{phase} F1-score (isFoggy is True)' : f1_isFoggyIsTrue,
                'n_epoch' : epoch})
            # log n_epoch and n_batch_iteration
            """
            acc, prec_isFoggyIsFalse, rec_isFoggyIsFalse, f1_isFoggyIsFalse = get_and_print_stats(confmat=cm_tot, mode=phase, label_isFoggy=0)
            wandb.log({f'{phase} precision (isFoggy is False)' : prec_isFoggyIsFalse})
            wandb.log({f'{phase} recall (isFoggy is False)' : rec_isFoggyIsFalse})            
            wandb.log({f'{phase} F1-score (isFoggy is False)' : f1_isFoggyIsFalse})
            """

        print()

    time_end = time()
    time_elapsed = time_end - time_start
    print(f'training and validation completed in {time_elapsed} seconds.')

    # saving model
    torch.save(obj=model, f=PATH_MODEL)


# REPRODUCIBILITY 
random_seed = 42

torch.seed()  # only for CPU
torch.manual_seed(random_seed)  # works for CPU and CUDA

# Fix all random seeds
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(random_seed)


############ ARGPARSERS, GLOBAL VARIABLES ############

parser = argparse.ArgumentParser(description='Run pretrained finetuning.')
parser.add_argument('--batch_size', type=int, help='batch size')
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--epochs', type=int, help='number of training epochs')
parser.add_argument('--train_split', type=float, help='train split')
parser.add_argument('--stations_cam', help='list of stations with camera number, separated with underscore (e.g. Buelenberg_1')
parser.add_argument('--weighted', help='how to weight the classes (manual: as given in script / Auto: Inversely proportional to occurance / False: not at all')
parser.add_argument('--path_dset', help='path to used dataset ')
parser.add_argument('--lr_scheduler', help='whether to use a lr scheduler')

args = parser.parse_args()

# logging
wandb.init(project="isFoggy_pretrained_finetuning", entity="jbaumer", config=args)

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
EPOCHS = args.epochs
TRAIN_SPLIT = args.train_split
WEIGHTED = args.weighted
PATH_DATASET = args.path_dset
LR_SCHEDULER = args.lr_scheduler

STATIONS_CAM_STR = args.stations_cam
STATIONS_CAM_STR = STATIONS_CAM_STR.replace("\\", "")
STATIONS_CAM_LST = sorted(ast.literal_eval(STATIONS_CAM_STR))  # sort to make sure not two models with data from same cameras (but input in different order) will be saved

N_CLASSES = 2
PATH_MODEL = f'models/{STATIONS_CAM_LST}_bs_{BATCH_SIZE}_LR_{LEARNING_RATE}_epochs_{EPOCHS}_weighted_{WEIGHTED}_lr_sched_{LR_SCHEDULER}'

# create datasets and dataloaders
dset = DischmaSet(root=PATH_DATASET, stat_cam_lst=STATIONS_CAM_LST)
print(f'Dischma set with data from {STATIONS_CAM_LST} created.')
dset_train, dset_val = get_train_val_split(dset)  # get e.g. 1 year of train data (eg 2019) and 4 mths of val data (e.g. 2020 Jan/April/July/Oct) - this val set must be handlabeled
len_dset_train, len_dset_val = len(dset_train), len(dset_val)
dloader_train = DataLoader(dataset=dset_train, batch_size=BATCH_SIZE)
dloader_val = DataLoader(dataset=dset_val, batch_size=BATCH_SIZE)

# class 0: is not foggy
# class 1: is foggy
n_class_0, n_class_1 = dset.get_balancedness()
n_tot = n_class_0 + n_class_1
w0, w1 = n_class_1/n_tot, n_class_0/n_tot

if WEIGHTED == 'False':
    weights = None
elif WEIGHTED == 'Manual':
    weights = torch.Tensor([0.3, 0.7]).to(device)  # w0 smaller, w1 larger because we want a high recall (only few FN) - when we predict a negative, we must be sure that it is negative (sunny)
elif WEIGHTED == 'Auto':
    weights = torch.Tensor([w0, w1]).to(device)


#if os.path.exists(PATH_MODEL) == True:
#    print('trained model already exists, loading model...')
#    model = torch.load(PATH_MODEL)
#else:
model = models.resnet18(pretrained=True)

criterion = nn.CrossEntropyLoss(reduction='mean', weight=weights)  # TODO: currently, all occurances are considered, optimal would be to only consider occ. of train split
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # TODO ev add momentum

if LR_SCHEDULER != 'None':
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=int(LR_SCHEDULER), gamma=0.1)  # Decay LR by a factor of 0.1 every 'step_size' epochs
elif LR_SCHEDULER == 'None':
    exp_lr_scheduler = None

# train all layers (should already be default)
for param in model.parameters():
    param.requires_grad = True
    param = param.to(device)  # prob not needed (whole model set to device later)

## sth like this for feature extractor
"""
# do not apply backprop on weight that were used for feature extraction
for param in model.parameters():
    param.requires_grad = False
    param = param.to(device)
"""

# adapt fully connected layer
n_features = model.fc.in_features
model.fc = nn.Linear(n_features, N_CLASSES)

# note: Softmax (from real to probab) is implicitly applied when working with crossentropyloss

model = model.to(device)
train_val_model(model=model, criterion=criterion, optimizer=optimizer, scheduler=exp_lr_scheduler, num_epochs=EPOCHS)
