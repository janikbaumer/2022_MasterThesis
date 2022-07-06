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
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

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

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

def print_grid(x, y, batchsize, batch_iteration):
    x = x.cpu()
    y = y.cpu().float()
    grid_img_x = torchvision.utils.make_grid(x, nrow=int(batchsize/2), normalize=True)
    grid_img_y = torchvision.utils.make_grid(y, nrow=int(batchsize/2), normalize=True)

    plt.figure()
    plt.title(f'batch iteration: {batch_iteration}')
    f, axarr = plt.subplots(2,1)
    axarr[0].imshow(grid_img_x.permute(1,2,0))
    axarr[0].set_title('Images')
    axarr[1].imshow(grid_img_y.permute(1,2,0))
    axarr[1].set_title('Labels')

    plt.savefig(f'stats/fig_check_manually/grid_segmentation_batch_iteration_{batch_iteration}')

def get_and_log_metrics(yt, ypred, ep, batch_it_loss, ph, bi=0):

    acc = accuracy_score(yt.cpu(), ypred.cpu())
    prec = precision_score(yt.cpu(), ypred.cpu())
    rec = recall_score(yt.cpu(), ypred.cpu())
    f1 = f1_score(yt.cpu(), ypred.cpu())
    # cm = confusion_matrix(y_true=yt.cpu(), y_pred=ypred.cpu())

    if LOGGING:
        wandb.log({
            f'{ph}/loss' : batch_it_loss,
            f'{ph}/accuracy' : acc,
            f'{ph}/precision' : prec,
            f'{ph}/recall' : rec,
            f'{ph}/F1-score' : f1,
            # f'{ph}/conf_mat' : wandb.plot.confusion_matrix(y_true=yt, preds=ypred, class_names=['class 0 (no snow)', 'class 1 (snow)']),  # this line uses extremely much CPU !!! - breaks program
            # f'{ph}/precision_recall_curve' : wandb.plot.pr_curve(y_true=yt, y_probas=yprob, labels=['class 0 (not foggy)', 'class 1 (foggy)']),
            'n_epoch' : ep,
            'batch_iteration' : bi})

    print(f'logged accuracy ({acc}), precision ({prec}), recall ({rec}) and f1 score ({f1})')

def test_model(model):
    pass

def train_val_model(model, criterion, optimizer, scheduler, num_epochs):
    time_start = time()

    batch_iteration = {}
    batch_iteration['train'] = 0
    batch_iteration['val'] = 0

    for epoch in range(num_epochs):
        print('\n', '-' * 10)

        for phase in ['train', 'val']:  # in each epoch, do training and validation
            print(f'{phase} phase in epoch {epoch+1}/{num_epochs} starting...')

            if phase == 'train':
                model.train()
                dloader = dloader_train

            else:
                model.eval()
                dloader = dloader_val

            # train_it_counter, val_it_counter = 0, 0
            running_loss = 0  # loss (to be updated during batch iteration)

            y_true_total = []
            y_pred_binary_total = []
            y_pred_logits_total = None

            y_true_total = torch.Tensor().to(device)  # initialize as empty tensor
            y_pred_binary_total = torch.Tensor().to(device)  # initialize as empty tensor


            for x, y in tqdm(dloader):
                batch_iteration[phase] += 1

                # print_grid(x,y, BATCH_SIZE, batch_iteration[phase])

                # move to GPU
                x = x.to(device)
                y = y.to(device)

                """
                if batch_iteration[phase] < 200 and batch_iteration[phase]%10 == 0:
                    print_grid(x,y, BATCH_SIZE, batch_iteration[phase])
                """

                # target only have one channels (containing the class indices) -> must be removed for loss function (if BCE loss)
                # # https://discuss.pytorch.org/t/only-batches-of-spatial-targets-supported-non-empty-3d-tensors-but-got-targets-of-size-1-1-256-256/49134
                y = y.squeeze()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    y_pred_logits = model(x)  # torch.Size([BS, n_classes, H, W]), e.g. [8, 3, 256, 256]  # logits, not probabilty !

                    # for metrics, get only argmax of first two cols (0 and 1), ignore 2nd col (with logits for no data)
                    y_pred_logits_data = y_pred_logits[:, 0:-1, :, :]  # consider all logits excepts for last class (no data) -> consider logits for class 0 and 1
                    y_pred_binary_data = y_pred_logits_data.argmax(axis=1, keepdim=False)  # only contains 0 or 1 -> predict always either 0 or 1

                    # y_pred_logits.shape: [BS, n_classes, H, W] // y.shape: [BS, H, W], contains values 0, 1, or 2 (vals in range n_classes)
                    loss = criterion(y_pred_logits, y)  # note: masking no data values is implicitly applied by setting weight for class 2 to zero 

                    if phase == 'train':
                        loss.backward()  # backprop
                        optimizer.step()  # update params

                # STATS
                y_true_flat = y.flatten()  # contains 0, 1, 2
                y_pred_flat_binary = y_pred_binary_data.flatten()  # contains 0, 1

                y_true_data = y_true_flat[y_true_flat != 2]  # contains 0, 1
                y_pred_data = y_pred_flat_binary[y_true_flat != 2]  # contains 0, 1

                y_true_total = torch.cat((y_true_total, y_true_data), 0)  # append to flattened torch tensor 
                y_pred_binary_total = torch.cat((y_pred_binary_total, y_pred_data), 0)

                # LOSSES
                batch_loss = loss.item() * x.shape[0]  # loss of whole batch (loss*batchsize (as loss was averaged ('mean')), each item of batch had this loss on avg)
                running_loss += batch_loss

                if phase == 'train':
                    if (batch_iteration[phase]%LOG_EVERY) == 0:
                        loss = running_loss/LOG_EVERY
                        print(f'batch iteration: {batch_iteration[phase]} / {len(dloader)*(epoch+1)} with {phase} loss (avg over these {LOG_EVERY} batch iterations): {loss}')
                        get_and_log_metrics(yt=y_true_total, ypred=y_pred_binary_total, ep=epoch, batch_it_loss=loss, ph=phase, bi=batch_iteration[phase])
                        running_loss = 0

                if phase == 'val':
                    if batch_iteration[phase]%len(dloader) == 0:
                        loss = running_loss / len(dloader)
                        print(f'batch iteration: {batch_iteration[phase]} / {len(dloader)*(epoch+1)} ... {phase} loss (avg over whole validation dataloader): {loss}')
                        get_and_log_metrics(yt=y_true_total, ypred=y_pred_binary_total, ep=epoch, batch_it_loss=loss, ph=phase, bi=batch_iteration[phase])
                        # as we're in last loop for validation, running_loss will be set to 0 anyways (changing the phase back to train)

            if phase == 'train':  # at end of epoch (training, could also be end of validation)
                if scheduler is not None:
                    scheduler.step()
                    # scheduler.print_lr()

        print()  # end of epoch e
    
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
if LOGGING:
    wandb.init(project="model_snow_segmentation", entity="jbaumer", config=args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # set device


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

N_CLASSES = 3
PATH_MODEL = f'models/segmentation/{STATIONS_CAM_LST}_bs_{BATCH_SIZE}_LR_{LEARNING_RATE}_epochs_{EPOCHS}_lr_sched_{LR_SCHEDULER}'
LOG_EVERY = 50


############ DATASETS AND DATALOADERS ############

# create datasets and dataloaders, split in train and validation set
dset_train = DischmaSet_segmentation(root=PATH_DATASET, stat_cam_lst=STATIONS_CAM_LST, mode='train')
dset_val = DischmaSet_segmentation(root=PATH_DATASET, stat_cam_lst=STATIONS_CAM_LST, mode='val')
dset_test = DischmaSet_segmentation(root=PATH_DATASET, stat_cam_lst=STATIONS_CAM_LST, mode='test')

# dset_full = DischmaSet_segmentation(root=PATH_DATASET, stat_cam_lst=STATIONS_CAM_LST, mode=None)
# dset_train, dset_val = random_split(dset_full, (int(len(dset_full)*TRAIN_SPLIT), math.ceil(len(dset_full)*(1-TRAIN_SPLIT))))
# print(f'Dischma sets (train and val) with data from {STATIONS_CAM_LST} created.')

dloader_train = DataLoader(dataset=dset_train, batch_size=BATCH_SIZE)
dloader_val = DataLoader(dataset=dset_val, batch_size=BATCH_SIZE)
dloader_test = DataLoader(dataset=dset_test, batch_size=BATCH_SIZE)

print('lengths (train, val, test dloader): ', len(dloader_train), len(dloader_val), len(dloader_test))

# balancedness: not done, as every image in label_path_list would have to be loaded and respective labels collected - takes much time and memory
"""
print('distribution of labels: ')
print(f'for train set: {dset_train.get_balancedness()}')
print(f'for val set: {dset_val.get_balancedness()}')
print(f'for test set: {dset_test.get_balancedness()}')
"""

# Note:
#   class 0: no data
#   class 1: snow
#   class 2/3: no snow

# TODO get balancedness of dset / ev consider weighting of classes (only of training data !)


############ MODEL, LOSS, OPTIMIZER, SCHEDULER  ############

model = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=N_CLASSES,              # model output channels (number of classes in your dataset)
)
model = model.to(device)

"""
# does not really work TODO ev delete
preprocess_input = get_preprocessing_fn('resnet34', pretrained='imagenet')
x_try = torch.randn(size=(8, 3, 512, 512), dtype=torch.float32)
x_try = preprocess_input(x_try)
out = model(x_try)
"""

# loss functions to try: BCE / IoU-loss / focal loss
criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, 1, 0]).to(device), reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)

if LR_SCHEDULER != 'None':
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=int(LR_SCHEDULER), gamma=0.1)  # Decay LR by a factor of 0.1 every 'step_size' epochs
elif LR_SCHEDULER == 'None':
    exp_lr_scheduler = None

train_val_model(model=model, criterion=criterion, optimizer=optimizer, scheduler=exp_lr_scheduler, num_epochs=EPOCHS)

test_model(model=model)