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
print('imports done')

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
    """
    yt, ypred: torch.Tensors with equal shapes
    """
    acc = accuracy_score(yt.cpu(), ypred.cpu())
    prec = precision_score(yt.cpu(), ypred.cpu())
    rec = recall_score(yt.cpu(), ypred.cpu())
    f1 = f1_score(yt.cpu(), ypred.cpu())
    cm = confusion_matrix(y_true=yt.cpu(), y_pred=ypred.cpu())

    if LOGGING:
        print(f'logging the {ph} set metrics...')

        if ph == 'train' or ph =='val' or ph == 'test':
            wandb.log({
                'n_epoch' : ep,
                'batch_iteration' : bi,
                f'{ph}/loss' : batch_it_loss,
                f'{ph}/accuracy' : acc,
                f'{ph}/precision' : prec,
                f'{ph}/recall' : rec,
                f'{ph}/F1-score' : f1,
                f'{ph}/cm' : cm,
                # f'{ph}/conf_mat' : wandb.plot.confusion_matrix(y_true=yt.cpu(), preds=ypred.cpu(), class_names=['class 0 (snow)', 'class 1 (no snow)']),  # this line uses extremely much CPU !!! - breaks program
                # f'{ph}/precision_recall_curve' : wandb.plot.pr_curve(y_true=yt, y_probas=yprob, labels=['class 0 (not foggy)', 'class 1 (foggy)']),
                })

    print(f'logged accuracy ({acc}), precision ({prec}), recall ({rec}) and f1 score ({f1})')

"""
def test_model(model):
    # with trained model, predict class for every x,y in test set
    # log the respective metrics (only one value per metric)
    # TODO: plot the incorrect classifications

    time_start = time()
    epoch = 0
    batch_iteration = {}
    batch_iteration['test'] = 0

    phase = 'test'
    print(f'{phase} phase starting...')
    model.eval()
    dloader = dloader_test

    running_loss = 0  # loss (to be updated during batch iteration)
    
    '''
    y_true_total = []
    y_pred_probab_total = None
    y_pred_logits_total = None
    y_pred_binary_total_th_std = []
    y_pred_binary_total_th_optimal = []
    '''

    y_true_total = []
    y_pred_binary_total = []
    y_pred_logits_total = None

    y_true_total = torch.Tensor().to(device)  # initialize as empty tensor
    y_pred_binary_total = torch.Tensor().to(device)  # initialize as empty tensor

    for x, y in dloader:
        batch_iteration[phase] += 1

        # move to GPU
        x = x.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(phase == 'train'):
            pred_logits = model(x)  # predictions (logits) (for class 0 and 1) / shape: batchsize, nclasses (8,2)
            loss = criterion(pred_logits, y)
            
            yprobab = torch.softmax(pred_logits, dim=1)  # [log_every*batchsize, 2] probas between 0 and 1, sum up to one
            #yprobab_neg = yprobab[:, 0]
            yprobab_pos = yprobab[:, 1]  # compute metrics for class one            
            
            threshold = torch.tensor([OPTIMAL_THRESHOLD]).to(device)

            pred_binary_th_std = yprobab.argmax(dim=1)  # threshold 0.5 # either 0 or 1 / shape: batchsize (8) / takes higher probablity (from the two classes) to compare to y (y_true)
            pred_binary_th_optimal = (yprobab_pos > threshold).float()  # threshold: last from validation # either 0 or 1 / shape: batchsize (8)



        # stats
        y_true = y.cpu().tolist()
        y_pred_binary_th_std = pred_binary_th_std.cpu().tolist()
        y_pred_binary_th_optimal = pred_binary_th_optimal.cpu().tolist()

        y_true_total.extend(y_true)
        y_pred_binary_total_th_std.extend(y_pred_binary_th_std)
        y_pred_binary_total_th_optimal.extend(y_pred_binary_th_optimal)

        if batch_iteration[phase] % len(dloader) != 1:
            y_pred_logits_total = torch.cat((y_pred_logits_total, pred_logits))
        else:
            y_pred_logits_total = pred_logits

        # losses
        batch_loss = loss.item() * x.shape[0]  # loss of whole batch (loss*batchsize (as loss was averaged ('mean')), each item of batch had this loss on avg)
        running_loss += batch_loss


        if batch_iteration[phase]%len(dloader) == 0:  # after having seen the whole test set, do logging
            loss = running_loss/len(dloader)
            print(f'batch iteration: {batch_iteration[phase]} / {len(dloader)*(epoch+1)} ... {phase} loss (avg over whole validation dataloader): {loss}')

            # TODO adapt function, log with std and optimal threshold
            get_and_log_metrics(yt=y_true_total, ypred_th_std=y_pred_binary_total_th_std, ylogits=y_pred_logits_total, ep=epoch, batch_it_loss=loss, ph=phase, bi=batch_iteration[phase], ypred_th_optimal=y_pred_binary_total_th_optimal)

    time_end = time()
    time_elapsed = time_end - time_start
    print(f'testing (on {device}) completed in {time_elapsed} seconds.')
"""

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

            #y_true_total = []
            #y_pred_binary_total = []
            #y_pred_logits_total = None

            y_true_total = torch.Tensor().to(device)  # initialize as empty tensor
            y_pred_binary_total = torch.Tensor().to(device)  # initialize as empty tensor


            for x, y in tqdm(dloader):
                # skip this iteration if only 0 (no_data) in the y tensor
                if np.unique(y.cpu()).shape == (1,) and np.unique(y.cpu())[0] == 0:
                    continue

                batch_iteration[phase] += 1
                
                # print_grid(x,y, BATCH_SIZE, batch_iteration[phase])

                # move to GPU, already done in getitem function of dataset class
                # x = x.to(device)
                # y = y.to(device)

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
                    # y_pred_logits.shape: [BS, n_classes, H, W]
                    # y.shape: [BS, H, W], contains values 0, 1, or 2 (vals in range n_classes)

                    loss = criterion(y_pred_logits, y)  # note: masking no data values is implicitly applied by setting weight for class 0 (no_data) to zero

                    # if loss or x or y contain nan elements, breakpoint

                    if phase == 'train':
                        loss.backward()  # backprop
                        optimizer.step()  # update params

                    # for metrics, get only argmax of cols 1 and 2, ignore col 0 (with logits for no data)
                    y_pred_logits_data = y_pred_logits[:, 1:, :, :]  # shape [BATCHSIZE, 2, 256, 256], consider all logits excepts for 0th class (no data) -> consider logits for class 1 and 2 (3)
                    y_pred_binary_data = y_pred_logits_data.argmax(axis=1, keepdim=False)  # only contains 0 (snow) or 1 (no_snow), note: this unfortunately also works if nan values in y_pred_logits_data 

                # STATS
                y_true_flat = y.flatten()  # contains 0, 1, 2
                # ev todo: log logits as well (not only predictions)
                # y_pred_flat_logits = y_pred_logits_data.view(BATCH_SIZE,2,-1)  # shape [BATCHSIZE, 2, 256x256], not completely flat
                y_pred_flat_binary = y_pred_binary_data.flatten()  # contains 0, 1

                # for metrics, only consider predictions of pixels that are not no_data
                y_true_data = y_true_flat[y_true_flat != 0]  # contains 1, 2
                y_true_data[y_true_data==1] = 0  # convert ones to zeros
                y_true_data[y_true_data==2] = 1  # convert twos to ones     # now contains 0 (snow), 1 (no_snow)
                y_pred_data = y_pred_flat_binary[y_true_flat != 0]  # contains 0, 1

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

        print()  # end of epoch
    
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
parser.add_argument('--optim', help='set type of optimizer (Adam or SGD)')
parser.add_argument('--weight_decay', type=float, help='set weight decay (used for Adam and for SGD')
parser.add_argument('--momentum', type=float, help='set momentum used for SGD optimizer')

args = parser.parse_args()

LOGGING = True
if LOGGING:
    wandb.init(project="model_snow_segmentation", entity="jbaumer", config=args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # set device


############ GLOBAL VARIABLES ############

BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
EPOCHS = args.epochs
TRAIN_SPLIT = args.train_split
PATH_DATASET = args.path_dset
LR_SCHEDULER = args.lr_scheduler
OPTIM = args.optim
WEIGHT_DECAY = args.weight_decay
MOMENTUM = args.momentum
# WEIGHTED = args.weighted

STATIONS_CAM_STR = args.stations_cam
STATIONS_CAM_STR = STATIONS_CAM_STR.replace("\\", "")
STATIONS_CAM_LST = sorted(ast.literal_eval(STATIONS_CAM_STR))  # sort to make sure not two models with data from same cameras (but input in different order) will be saved

N_CLASSES = 3
PATH_MODEL = f'models/segmentation/{STATIONS_CAM_LST}_bs_{BATCH_SIZE}_LR_{LEARNING_RATE}_epochs_{EPOCHS}_lr_sched_{LR_SCHEDULER}'
LOG_EVERY = 200


############ DATASETS AND DATALOADERS ############

# create datasets and dataloaders, split in train and validation set
dset_train = DischmaSet_segmentation(root=PATH_DATASET, stat_cam_lst=STATIONS_CAM_LST, mode='train')
dset_val = DischmaSet_segmentation(root=PATH_DATASET, stat_cam_lst=STATIONS_CAM_LST, mode='val')
dset_test = DischmaSet_segmentation(root=PATH_DATASET, stat_cam_lst=STATIONS_CAM_LST, mode='test')

# dset_full = DischmaSet_segmentation(root=PATH_DATASET, stat_cam_lst=STATIONS_CAM_LST, mode=None)
# dset_train, dset_val = random_split(dset_full, (int(len(dset_full)*TRAIN_SPLIT), math.ceil(len(dset_full)*(1-TRAIN_SPLIT))))
# print(f'Dischma sets (train and val) with data from {STATIONS_CAM_LST} created.')

dloader_train = DataLoader(dataset=dset_train, batch_size=BATCH_SIZE, shuffle=True)
dloader_val = DataLoader(dataset=dset_val, batch_size=BATCH_SIZE)
dloader_test = DataLoader(dataset=dset_test, batch_size=BATCH_SIZE)

print('lengths (train, val, test dloader): ', len(dloader_train), len(dloader_val), len(dloader_test))

# balancedness: not done, as every image in label_path_list would have to be loaded and respective labels collected - takes (too) much time and memory
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
#model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
#    in_channels=3,
#    out_channels=N_CLASSES,
#    init_features=32,
#    pretrained=True)

model = model.to(device)
model = model.to(torch.float64)  # set higher precision

"""
# does not really work TODO ev delete
preprocess_input = get_preprocessing_fn('resnet34', pretrained='imagenet')
x_try = torch.randn(size=(8, 3, 512, 512), dtype=torch.float32)
x_try = preprocess_input(x_try)
out = model(x_try)
"""

# loss functions to try: BCE / IoU-loss / focal loss
criterion = nn.CrossEntropyLoss(weight=torch.Tensor([0, 1, 1]).to(device).to(torch.float64), reduction='mean')

if OPTIM == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
elif OPTIM == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)  # weight_decay: It is used for adding the l2 penality to the loss (default = 0)

if LR_SCHEDULER != 'None':
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=int(LR_SCHEDULER), gamma=0.8)  # Decay LR by a factor of gamma every 'step_size' epochs
elif LR_SCHEDULER == 'None':
    exp_lr_scheduler = None

train_val_model(model=model, criterion=criterion, optimizer=optimizer, scheduler=exp_lr_scheduler, num_epochs=EPOCHS)

# test_model(model=model)
