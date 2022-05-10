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

def print_grid(x, y, batchsize, batch_iteration):
    x = x.cpu()
    y = y.cpu()
    # print(y)
    y_reshaped = y.reshape(2, -1).numpy()
    grid_img = torchvision.utils.make_grid(x, nrow=int(batchsize/2), normalize=True)
    plt.title(f'batch iteration: {batch_iteration}\n{y_reshaped[0]}\n{y_reshaped[1]}')
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.savefig(f'stats/fig_check_manually/grid_batch_iteration_{batch_iteration}')


def train_val_model(model, criterion, optimizer, scheduler, num_epochs):
    time_start = time()

    batch_iteration = {}
    batch_iteration['train'] = 0
    batch_iteration['val'] = 0

    for epoch in range(num_epochs):
        print('\n', '-' * 10)

        for phase in ['train', 'val']:  # in each epoch, do training and validation
            print(f'{phase} phase in epoch {epoch+1}/{num_epochs} starting...')

            train_it_counter, val_it_counter = 0, 0
            running_loss = 0
            batch_it_loss, epoch_loss = 0, 0

            y_true_total = []
            y_pred_probab_total = []
            y_pred_binary_total = []
    
            if phase == 'train':
                model.train()
                dloader = dloader_train
                dset = dset_train

            else:
                model.eval()
                dloader = dloader_val
                dset = dset_val


            for x, y in dloader_train:
                batch_iteration[phase] += 1
                #save_x_y(x, y)

                x = x.to(device)
                y = y.to(device)

                x = x.float()
                y = y.long()

                """
                # plot some batches
                #if batch_iteration < 200 and batch_iteration%10 == 0:
                #    print_grid(x,y, BATCH_SIZE, batch_iteration)
                #if batch_iteration == 0:
                #    print_grid(x,y, BATCH_SIZE, batch_iteration)
                """

                norm = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                x = norm(x)

                # If your targets contain the class indices already, you should remove the channel dimension:
                # https://discuss.pytorch.org/t/only-batches-of-spatial-targets-supported-non-empty-3d-tensors-but-got-targets-of-size-1-1-256-256/49134
                y = y.squeeze()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    y_pred_logits = model(x)  # torch.Size([8, 3, 256, 256])  # logits, not probabilty !
                    y_pred_binary = y_pred_logits.argmax(axis=1, keepdim=False)  # for each batch image, choose class with highest probability

                    loss = criterion(y_pred_logits, y)  # note: masking no data values is implicitly applied by setting weight for class 2 to zero 

                    if phase == 'train':
                        loss.backward()  # backprop
                        optimizer.step()  # update params

                # TODO:
                """
                get ytrue, ypred, ypredbinary (all as lists/arrays) and add (extend) to total list/arr -> evaluate metrics
                """
                # STATS
                y_true_lst = list(y.flatten().cpu().numpy())
                y_pred_binary_lst = list(y_pred_binary.flatten().cpu().numpy())

                y_true_arr = np.array(y.flatten().cpu().numpy())
                y_pred_binary_arr = np.array(y_pred_binary.flatten().cpu().numpy())


                y_true_total.extend(y_true_lst)
                # y_pred_probab_total.extend(...)
                y_pred_binary_total.extend(y_pred_binary_lst)

                # remove values where either no data in GT or no data was predicted

                # remove values in GT and in pred where the GT is no data
                # after this block, there will still be no data values in yp_data_in_GT
                yt_data_in_GT = y_true_arr[y_true_arr != 2]
                yp_data_in_GT = y_pred_binary_arr[y_true_arr != 2]

                # remove values in GT and in pred where the prediction is no data (resp. where prediction was 2)
                yt_data_only = yt_data_in_GT[yp_data_in_GT != 2]
                yp_data_only = yp_data_in_GT[yp_data_in_GT != 2]
                # TODO calculate metrics from yt_data and yp_data

                yt = yt_data_only
                yp = yp_data_only

                # LOSSES
                batch_loss = loss.item() * x.shape[0]  # loss of whole batch (loss*batchsize (as loss was averaged ('mean')), each item of batch had this loss on avg)
                running_loss += batch_loss

                if phase == 'train':
                    train_it_counter += 1
                    LOG_EVERY = 50
                    if (batch_iteration[phase]%LOG_EVERY) == 0:
                        loss = running_loss/LOG_EVERY
                        print(f'batch iteration: {batch_iteration[phase]} / {len(dloader)*(epoch+1)} with {phase} loss (avg over these {LOG_EVERY} batch iterations): {loss}')

                        # TODO: get some metrics
                        # TODO: log some metrics
                        """
                        acc, prec, rec, f1 = get_and_log_metrics(yt=y_true_total, ypred=y_pred_binary_total, yprob=y_pred_probab_total, ep=epoch, batch_it_loss=loss, ph=phase, bi=batch_iteration[phase])
                        print(f'logged accuracy ({acc}), precision ({prec}), recall ({rec}) and f1 score ({f1})')
                        """

                        running_loss = 0

                if phase == 'val':
                    val_it_counter += 1

                    if batch_iteration[phase]%len/(dloader) == 0:
                        loss = running_loss / len(dloader)
                        print(f'batch iteration: {batch_iteration[phase]} / {len(dloader)*(epoch+1)} ... {phase} loss (avg over whole validation dataloader): {loss}')

                        # TODO: get some metrics
                        # TODO: log some metrics

                        """
                        acc, prec, rec, f1 = get_and_log_metrics(yt=y_true_total, ypred=y_pred_binary_total, yprob=y_pred_probab_total, ep=epoch, batch_it_loss=loss, ph=phase, bi=batch_iteration[phase])
                        print(f'logged accuracy ({acc}), precision ({prec}), recall ({rec}) and f1 score ({f1})')
                        """

            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()
                    # scheduler.print_lr()
            
            epoch_loss = running_loss / len(dset)
            print(f'epoch loss in {phase} phase: {epoch_loss}')

        print()
    
    time_end = time()
    time_elapsed = time_end - time_start

    print(f'training and validation (on {device}) completed in {time_elapsed} seconds.')

    """
    # saving model
    torch.save(obj=model, f=PATH_MODEL)
    """

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
    classes=3,                      # model output channels (number of classes in your dataset)
)
model = model.to(device)

"""
preprocess_input = get_preprocessing_fn('resnet34', pretrained='imagenet')
x_try = torch.randn(size=(8, 3, 512, 512), dtype=torch.float32)
x_try = preprocess_input(x_try)
out = model(x_try)
"""

# loss functions to try: BCE / IoU-loss / focal loss
criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, 1, 0]).to(device), reduction='mean')  # TODO: currently, all occurances are considered, optimal would be to only consider occ. of train split
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)  # TODO ev add momentum

exp_lr_scheduler = None

print()

train_val_model(model=model, criterion=criterion, optimizer=optimizer, scheduler=exp_lr_scheduler, num_epochs=EPOCHS)

