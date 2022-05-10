import torch
import torchvision
import argparse
import matplotlib.pyplot as plt
import os
import wandb
import ast
import random

import numpy as np
from time import time
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score


from torch.utils.data import DataLoader, random_split
from torch import nn
from torchvision import models
from torch.optim import lr_scheduler

from dischma_set_classification import DischmaSet_classification

print('imports done')

# note: class 1 = foggy images

# TODO:
    # Log confusion matrix
    # validation more often (e.g. every 200th loop) on whole dloader_val # see: (Pytorch Lightning limit_val_batches and val_check_interval behavior)
    # if other models, use BS small enogh s.t. it works
    # precision recall curve - if looks strange, ev used metrics needs to be changed (e.g. to optimal F1 score, as it's robust to shifted PR curves (if they are shifted to one side))

# cuda problems model (batch size, ...?)
# reset batch_iteration to 0 (every step in dloader)
# validation results meaningful?


# TODO (after meeting):
    # check data augmentation (horizontal flipping, cropping to eg 80-90%)), ev more
        # issue found: mean of image net could not be used for our data (mean was not 0 after tf)
        # TODO easier solution: make sure imgs are in range (0,1) not (0.255) -> check mean and std
        # assumption: learned features from resnet can also be used for our imgaes (same pixel distribution)
        # (might be slightly wrong, as I am using imgs with more snow/fog than normal (imagenet) - mean and std will be slightly off (not 0, resp 1))

        # solution: calculate mean every time and for each image to be computed ->
        #   newtf = transforms.Normalize((image.mean((1,2))), (0.229, 0.224, 0.225)) TODO: same thing for std
        #   newimg = newtf(oldimg)
        #  check: newimg.mean() should be ~0 
    # also do data augmentation on validation set (then take average, or consider as foggy if at least once foggy)
    # only take one model that works, do not play around too much
    # check variable batch iteration (check how often model will be evaluated)
    # validate/train model - evaluate with only handlabelled data
    # add confusion matrix to wandb.log
    # add precision recall curve to wandb.log
    # train and validate (with handlabelled data) - get metrics (wandb)
    # train and validate (with handlabelled data, multiple cams) - get metrics (wandb)
    # send selection of downsampled data / adapted script handlabelling.py to ehafner (to manually label data)

################# FUNCTIONS ######################

def get_and_log_metrics(yt, ypred, yprob, ep, batch_it_loss, ph, bi=0):
    acc = accuracy_score(y_true=yt, y_pred=ypred)
    prec = precision_score(y_true=yt, y_pred=ypred)
    rec = recall_score(y_true=yt, y_pred=ypred)
    f1 = f1_score(y_true=yt, y_pred=ypred)
    cm = confusion_matrix(y_true=yt, y_pred=ypred)

    wandb.log({
        f'{ph}/loss' : batch_it_loss,
        f'{ph}/accuracy' : acc,
        f'{ph}/precision (isFoggy is True)' : prec,
        f'{ph}/recall (isFoggy is True)' : rec,  # this should be high !!! (to catch all (foggy) images)
        f'{ph}/F1-score (isFoggy is True)' : f1,
        f'{ph}/conf_mat' : wandb.plot.confusion_matrix(y_true=yt, preds=ypred, class_names=['class 0 (not foggy)', 'class 1 (foggy)']),
        # f'{ph}/precision_recall_curve' : wandb.plot.pr_curve(y_true=yt, y_probas=yprob, labels=['class 0 (not foggy)', 'class 1 (foggy)']),
        'n_epoch' : ep,
        'batch_iteration' : bi})

    return acc, prec, rec, f1

def get_train_val_split(dset_full):
    print('splitting in train/test...')
    len_full = len(dset_full)
    len_train = int(TRAIN_SPLIT*len_full)
    len_val = len_full - len_train
    dset_train, dset_val = random_split(dset_full, [len_train, len_val])  # Split Pytorch tensor
    return dset_train, dset_val

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

    """
    batch_iteration = 0
    """
    
    batch_iteration = {}
    batch_iteration['train'] = 0
    batch_iteration['val'] = 0
    
    for epoch in range(num_epochs):
        print('\n', '-' * 10)
        


        for phase in ['train', 'val']:  # in each epoch, do training and validation
            print(f'{phase} phase in epoch {epoch+1}/{num_epochs} starting...')

            train_it_counter, val_it_counter = 0, 0

            if phase == 'train':
                model.train()
                dloader = dloader_train
                dset = dset_train
                """
                len_dset_current = len(dset_train)
                """
            else:
                model.eval()
                dloader = dloader_val
                dset = dset_val
                """
                len_dset_current = len(dset_val)
                """

            running_loss = 0  # loss (to be updated during batch iteration)
            batch_it_loss, epoch_loss = 0, 0

            """
            y_true_batch_it, y_pred_batch_it, y_probab_batch_it = [], [], []
            # cm_tot = np.zeros((2,2), dtype='int64')
            # y_true_total, y_pred_total = [], []
            """
            y_true_total = []
            y_pred_probab_total = []
            y_pred_binary_total = []


            for x, y in dloader:
                """
                batch_iteration += 1
                """
                batch_iteration[phase] += 1


                # move to GPU
                x = x.to(device)
                y = y.to(device)

                """
                # plot some batches
                #if batch_iteration < 200 and batch_iteration%10 == 0:
                #    print_grid(x,y, BATCH_SIZE, batch_iteration)
                #if batch_iteration == 0:
                #    print_grid(x,y, BATCH_SIZE, batch_iteration)
                """

                # zero the parameter gradients (not keep them from last batch)
                optimizer.zero_grad()

                # forward pass, track history (calc gradients) only in training
                with torch.set_grad_enabled(phase == 'train'):
                    pred = model(x)  # probabilities (for class 0 and 1) / shape: batchsize, nclasses (8,2)
                    y_probab = pred[:,1]   # probability for class one / shape: batchsize (8) / vals in range (0,1)
                    pred_binary = pred.argmax(dim=1)   # either 0 or 1 / shape: batchsize (8) / take higher value (from the two classes) to compare to y (y_true) # TODO: maybe use different thresholds
                    loss = criterion(pred, y)

                    if phase == 'train':
                        loss.backward()  # backprop
                        optimizer.step()  # update params

                # stats
                """
                y_true = y.cpu().tolist()
                y_probab = y_probab.cpu().tolist()
                y_pred = pred_binary.cpu().tolist()
                """
                predictions = pred.cpu().detach().numpy()
                y_true = y.cpu().tolist()
                y_pred_probab = y_probab.cpu().tolist()  # prob for class one
                y_pred_binary = pred_binary.cpu().tolist()

                y_true_total.extend(y_true)
                y_pred_probab_total.extend(y_pred_probab)
                y_pred_binary_total.extend(y_pred_binary)

                # losses
                batch_loss = loss.item() * x.shape[0]  # loss of whole batch (loss*batchsize (as loss was averaged ('mean')), each item of batch had this loss on avg)
                running_loss += batch_loss

                """
                # get losses
                batch_loss = loss.item() * x.size(0)  # loss of whole batch, loss*batchsize (as loss was averaged ('mean')), each item of batch had this loss on avg
                running_loss += batch_loss
                batch_it_loss += batch_los


                y_true_batch_it.extend(y_true)
                y_probab_batch_it.extend(y_probab)
                y_pred_batch_it.extend(y_pred)

                # y_true_total.extend(y_true)
                # y_pred_total.extend(y_pred)

                # array([[a, b],
                #        [c, d]])
                # a: tn (0 true, 0 predicted)  # negative is predicted, and the prediction is true
                # b: fp (0 true, 1 predicted)  # positive is predicted, and the prediction is wrong
                # c: fn (1 true, 0 predicted)  # negative is predicted, and the prediction is wrong
                # d: tp (1 true, 1 predicted)  # positive is predicted, and the prediction is true
                # to extract all values:
                # (tn, fp, fn, tp) = cm.ravel()

                ##### cm_current = confusion_matrix(y_true, y_pred)
                ##### cm_tot = cm_tot + cm_current  # accumulate confusion matrices

                # batch_iteration += 1

                """

                if phase == 'train':
                    train_it_counter += 1
                    LOG_EVERY = 200
                    if (batch_iteration[phase]%LOG_EVERY) == 0:
                        """
                        batch_it_loss = batch_it_loss/LOG_EVERY
                        """
                        loss = running_loss/LOG_EVERY
                        """print(f'batch iteration: {batch_iteration} / {len(dloader)} ... batch loss ({phase}), avg over these 200 batch iterations: ', batch_it_loss)"""
                        print(f'batch iteration: {batch_iteration[phase]} / {len(dloader)*(epoch+1)} with {phase} loss (avg over these {LOG_EVERY} batch iterations): {loss}')

                        # get metrics
                        """acc, prec, rec, f1 = get_and_log_metrics(yt=y_true_batch_it, ypred=y_pred_batch_it, yprob=y_probab_batch_it, ep=epoch, batch_it_loss=batch_it_loss, ph=phase, bi=batch_iteration)"""
                        acc, prec, rec, f1 = get_and_log_metrics(yt=y_true_total, ypred=y_pred_binary_total, yprob=y_pred_probab_total, ep=epoch, batch_it_loss=loss, ph=phase, bi=batch_iteration[phase])
                        print(f'logged accuracy ({acc}), precision ({prec}), recall ({rec}) and f1 score ({f1})')

                        running_loss = 0

                        """
                        batch_it_loss = 0
                        y_true_batch_it, y_pred_batch_it = [], []
                        """


                if phase == 'val':
                    val_it_counter += 1

                    if batch_iteration[phase]%len(dloader) == 0:
                        """batch_it_loss = batch_it_loss/len(dloader)"""
                        loss = running_loss/len(dloader)
                        print(f'batch iteration: {batch_iteration[phase]} / {len(dloader)*(epoch+1)} ... {phase} loss (avg over whole validation dataloader): {loss}')
                        # get metrics
                        """acc, prec, rec, f1 = get_and_log_metrics(yt=y_true_batch_it, ypred=y_pred_batch_it, yprob=y_probab_batch_it, ep=epoch, batch_it_loss=batch_it_loss, ph=phase, bi=batch_iteration)"""
                        acc, prec, rec, f1 = get_and_log_metrics(yt=y_true_total, ypred=y_pred_binary_total, yprob=y_pred_probab_total, ep=epoch, batch_it_loss=loss, ph=phase, bi=batch_iteration[phase])
                        print(f'logged accuracy ({acc}), precision ({prec}), recall ({rec}) and f1 score ({f1})')

            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()
                    # scheduler.print_lr()


            epoch_loss = running_loss / len(dset)
            print(f'epoch loss in {phase} phase: {epoch_loss}')

            """
            # epoch_loss = running_loss/len_dset_current
            epoch_loss = running_loss/(len(dloader_train) + len(dloader_val))
            print('epoch loss = ', epoch_loss)
            """

            """
            #acc, prec_isFoggyIsTrue, rec_isFoggyIsTrue, f1_isFoggyIsTrue = get_and_print_stats(confmat=cm_tot, mode=phase, label_isFoggy=1)
            # TODO: do every 200th batch iteration or so
            #wandb.log({
            #    f'{phase} loss' : epoch_loss,
            #    f'{phase} accuracy' : acc,
            #    f'{phase} precision (isFoggy is True)' : prec_isFoggyIsTrue,
            #    f'{phase} recall (isFoggy is True)' : rec_isFoggyIsTrue,  # this should be high !!! (to catch all (foggy) images)
            #    f'{phase} F1-score (isFoggy is True)' : f1_isFoggyIsTrue,
            #    'n_epoch' : epoch})
            # log n_epoch and n_batch_iteration
            acc, prec_isFoggyIsFalse, rec_isFoggyIsFalse, f1_isFoggyIsFalse = get_and_print_stats(confmat=cm_tot, mode=phase, label_isFoggy=0)
            wandb.log({f'{phase} precision (isFoggy is False)' : prec_isFoggyIsFalse})
            wandb.log({f'{phase} recall (isFoggy is False)' : rec_isFoggyIsFalse})
            wandb.log({f'{phase} F1-score (isFoggy is False)' : f1_isFoggyIsFalse})
            """

        print()

    time_end = time()
    time_elapsed = time_end - time_start
    print(f'training and validation (on {device}) completed in {time_elapsed} seconds.')

    # saving model
    torch.save(obj=model, f=PATH_MODEL)


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

# logging
wandb.init(project="isFoggy_pretrained_finetuning", entity="jbaumer", config=args)

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############ GLOBAL VARIABLES ############

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
###dset(_full) = DischmaSet_classification(root=PATH_DATASET, stat_cam_lst=STATIONS_CAM_LST)
dset_train = DischmaSet_classification(root=PATH_DATASET, stat_cam_lst=STATIONS_CAM_LST, mode='train')
dset_val = DischmaSet_classification(root=PATH_DATASET, stat_cam_lst=STATIONS_CAM_LST, mode='val')
print(f'Dischma sets (train and val) with data from {STATIONS_CAM_LST} created.')

# dset_train, dset_val = get_train_val_split(dset)  # get e.g. 1 year of train data (eg 2020) and 4 mths of val data (e.g. 2021 Jan/April/July/Oct) - this val set must be handlabeled

len_dset_train, len_dset_val = len(dset_train), len(dset_val)

"""
# to try with only handlabeled data
dset_full = DischmaSet_classification(root=PATH_DATASET, stat_cam_lst=STATIONS_CAM_LST, mode='val')
dset_train, dset_val = get_train_val_split(dset_full)  # get e.g. 1 year of train data (eg 2020) and 4 mths of val data (e.g. 2021 Jan/April/July/Oct) - this val set must be handlabeled
"""

dloader_train = DataLoader(dataset=dset_train, batch_size=BATCH_SIZE)
dloader_val = DataLoader(dataset=dset_val, batch_size=BATCH_SIZE)

# class 0: is not foggy
# class 1: is foggy
n_class_0, n_class_1 = dset_train.get_balancedness()
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

# dict(model.named_modules()) -> gets all layers with implementation (only names: dct.keys() )



### RESNET -- should work
model = models.resnet18(pretrained=True)
n_features = model.fc.in_features  # adapt fully connected layer
model.fc = nn.Linear(n_features, N_CLASSES)  # note: Softmax (from real to probab) is implicitly applied when working with crossentropyloss

# ### EFFICIENT NET -- too much memory used
# model = models.efficientnet_b1(pretrained=True)
# n_features = model.classifier[1].in_features  # adapt fully connected layer
# model.classifier[1] = nn.Linear(n_features, N_CLASSES)
# 
# ### DENSE NET -- too much memory used
# model = models.densenet121()
# n_features = model.classifier.in_features
# model.classifier = nn.Linear(n_features, N_CLASSES)
# 
# VGG 11 -- should work

# model = models.vgg11()
# f_in = model.classifier[6].out_features
# model.classifier.add_module('new_relu', torch.nn.modules.activation.ReLU(inplace=True))  # add to append to last sequential block (of the classifier)
# model.classifier.add_module('new_dropout', torch.nn.modules.dropout.Dropout(p=0.5, inplace=False))
# model.classifier.add_module('new_linear', torch.nn.modules.Linear(in_features=f_in, out_features=N_CLASSES))

# model = models.GoogLeNet(num_classes=N_CLASSES)

model = model.to(device)

# note: Softmax (from real to probab) is implicitly applied when working with crossentropyloss
criterion = nn.CrossEntropyLoss(reduction='mean', weight=weights)  # TODO: currently, all occurances are considered, optimal would be to only consider occ. of train split
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)  # TODO ev add momentum

if LR_SCHEDULER != 'None':
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=int(LR_SCHEDULER), gamma=0.1)  # Decay LR by a factor of 0.1 every 'step_size' epochs
elif LR_SCHEDULER == 'None':
    exp_lr_scheduler = None

"""
# train all layers (should already be default)
for param in model.parameters():
    param.requires_grad = True  # if False, do not apply backprop on weight that were used for feature extraction -> fixed feature extractor
    param = param.to(device)  # prob not needed (whole model set to device later)
"""

train_val_model(model=model, criterion=criterion, optimizer=optimizer, scheduler=exp_lr_scheduler, num_epochs=EPOCHS)

