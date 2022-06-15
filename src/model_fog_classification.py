from re import T
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
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, precision_recall_curve

from torch.utils.data import DataLoader, random_split
from torch import nn
from torchvision import models
from torch.optim import lr_scheduler
from tqdm import tqdm
from dischma_set_classification import DischmaSet_classification

print('imports done')


# NOTES:
    # class 1 = foggy images
    # if using other models, use BS small enogh s.t. it works
    # assumption: learned features from resnet can also be used for our images (same pixel distribution)
    # (might be slightly wrong, as we are using imgs with more snow/fog than normal (imagenet) - mean and std will be slightly off (not 0, resp 1))
    # only take one model that works, do not play around too much
    # validate/train model - evaluate with only handlabelled data

# multiple TODO s: notes (after meeting):
    # precision recall curve - if looks strange, ev used metrics needs to be changed (e.g. to optimal F1 score, as it's robust to shifted PR curves (if they are shifted to one side))
    # check data augmentation manually (horizontal flipping, cropping to eg 80-90%)), ev more
    # ev also do data augmentation on validation set (then take average, or consider as foggy if at least once foggy)
    # add precision recall curve to wandb.log
    # train and validate (with handlabelled data) - get metrics (wandb)
    # train and validate (with handlabelled data, multiple cams) - get metrics (wandb)


################# FUNCTIONS ######################

def get_optimal_f1_score():
    pass

def get_best_threshold():
    # from vector of predictions, get best threshold
    pass

def get_balance(dset):
    lst = []
    for ele in dset:
        lst.append(dset[ele][1])


def get_and_log_metrics(yt, ypred, ylogits, ep, batch_it_loss, ph, bi=0):
    """
    yt, ypred: lists
    yprob: torch tensor
    """
    yprobab = torch.sigmoid(ylogits)  # [log_every*batchsize, 2] probas between 0 and 1
    yprobab_neg = yprobab[:, 0]
    yprobab_pos = yprobab[:, 1]  # compute precision and recall for class one
    acc = accuracy_score(y_true=yt, y_pred=ypred)
    prec = precision_score(y_true=yt, y_pred=ypred)
    rec = recall_score(y_true=yt, y_pred=ypred)
    f1 = f1_score(y_true=yt, y_pred=ypred)

    f1_opt = get_optimal_f1_score()
    precs, recs, Threshs = precision_recall_curve(yt, yprobab_pos.detach().cpu())  

    if LOGGING:
        table = wandb.Table(data=[[x, y] for (x, y) in zip(precs[::len(precs)//9800], recs[::len(recs)//9800])], columns = ["Precision", "Recall"]) 
        wandb.log({
            f'{ph}/loss' : batch_it_loss,
            f'{ph}/accuracy' : acc,
            f'{ph}/precision' : prec,
            f'{ph}/recall' : rec,  # this should be high !!! (to catch all foggy images)
            f'{ph}/F1-score' : f1,
            f'{ph}/conf_mat' : wandb.plot.confusion_matrix(y_true=yt, preds=ypred, class_names=['class 0 (not foggy)', 'class 1 (foggy)']),
            f'{ph}/precision_recall_curve' : wandb.plot.pr_curve(y_true=yt, y_probas=yprob.detach().cpu(), labels=['class 0 (not foggy)', 'class 1 (foggy)']),
            f'{ph}/PR_Curve' : wandb.plot.line(table, 'Precision', 'Recall', title='PR-Curve'),
            'n_epoch' : ep,
            'batch_iteration' : bi})

        print(f'logged accuracy ({acc}), precision ({prec}), recall ({rec}) and f1 score ({f1})')


def get_train_val_split(dset_full):
    print('splitting in train/test...')
    len_full = len(dset_full)
    len_train = int(TRAIN_SPLIT*len_full)
    len_val = len_full - len_train
    dset_train, dset_val = random_split(dset_full, [len_train, len_val])  # Split Pytorch tensor
    return dset_train, dset_val


def print_grid(x, y, batchsize, bi):
    x = x.cpu()
    y = y.cpu()
    # print(y)
    y_reshaped = y.reshape(2, -1).numpy()
    grid_img = torchvision.utils.make_grid(x, nrow=int(batchsize/2), normalize=True)
    plt.title(f'batch iteration: {bi}\n{y_reshaped[0]}\n{y_reshaped[1]}')
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.savefig(f'stats/fig_check_manually/grid_batch_iteration_{bi}')


def test_model(model):
    # with trained model, predict class for every x,y in test set
    # log the respective metrics (only one per metric)
    # plot the incorrect classifications

    time_start = time()
    epoch = 0
    batch_iteration = {}
    batch_iteration['test'] = 0

    phase = 'test'
    print(f'{phase} phase starting...')
    model.eval()
    dloader = dloader_test

    running_loss = 0  # loss (to be updated during batch iteration)
    y_true_total = []
    y_pred_probab_total = None
    y_pred_logits_total = None
    y_pred_binary_total = []

    for x, y in tqdm(dloader):
        batch_iteration[phase] += 1

        # move to GPU
        x = x.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(phase == 'train'):
            pred_logits = model(x)  # predictions (logits) (for class 0 and 1) / shape: batchsize, nclasses (8,2)
            pred_binary = pred_logits.argmax(dim=1)  # [8], threshold 0.5 # either 0 or 1 / shape: batchsize (8) / take higher value (from the two classes) to compare to y (y_true)
            loss = criterion(pred_logits, y)

        # stats
        y_true = y.cpu().tolist()
        y_pred_binary = pred_binary.cpu().tolist()

        y_true_total.extend(y_true)
        y_pred_binary_total.extend(y_pred_binary)

        # losses
        batch_loss = loss.item() * x.shape[0]  # loss of whole batch (loss*batchsize (as loss was averaged ('mean')), each item of batch had this loss on avg)
        running_loss += batch_loss


        if batch_iteration[phase]%len(dloader) == 0:  # after having seen the whole test set, do logging
            loss = running_loss/len(dloader)
            print(f'batch iteration: {batch_iteration[phase]} / {len(dloader)*(epoch+1)} ... {phase} loss (avg over whole validation dataloader): {loss}')

            get_and_log_metrics(yt=y_true_total, ypred=y_pred_binary_total, ylogits=y_pred_probab_total, ep=epoch, batch_it_loss=loss, ph=phase, bi=batch_iteration[phase])

    time_end = time()
    time_elapsed = time_end - time_start
    print(f'training and validation (on {device}) completed in {time_elapsed} seconds.')


def train_val_model(model, criterion, optimizer, scheduler, num_epochs):
    time_start = time()

    batch_iteration = {}
    batch_iteration['train'] = 0
    batch_iteration['val'] = 0

    for epoch in range(num_epochs):
        print('\n', '-' * 10)

        for phase in ['train', 'val']:  # in each epoch, do training and validation
            print()
            print(f'{phase} phase in epoch {epoch+1}/{num_epochs} starting...')

            if phase == 'train':
                model.train()
                dloader = dloader_train

            else:
                model.eval()
                dloader = dloader_val

            running_loss = 0  # loss (to be updated during batch iteration)

            y_true_total = []
            y_pred_probab_total = None
            y_pred_logits_total = None
            y_pred_binary_total = []


            for x, y in tqdm(dloader):
                #for i in range(1000):
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
                    pred_logits = model(x)  # predictions (logits) (for class 0 and 1) / shape: batchsize, nclasses (8,2)

                    y_pred_logits_neg = pred_logits[:, 0]  # [8]
                    y_pred_logits_pos = pred_logits[:, 1]  # [8]


                    # y_probab = pred_logits[:,1]   # probability for class one: pred[:,1] / shape: batchsize (8)
                    # y_probab = pred_logits

                    pred_binary = pred_logits.argmax(dim=1)  # [8], threshold 0.5 # either 0 or 1 / shape: batchsize (8) / take higher value (from the two classes) to compare to y (y_true)

                    loss = criterion(pred_logits, y)
                    if phase == 'train':
                        loss.backward()  # backprop
                        optimizer.step()  # update params

                # stats
                y_true = y.cpu().tolist()

                # y_pred_probab = y_probab.cpu().tolist()  # prob for class one
                # y_pred_probab = y_probab
                y_pred_binary = pred_binary.cpu().tolist() 

                # debug - problematic: y_pred_binary is full with 0 (1 never gets predicted)
                #print(y_true)
                #print(y_pred_binary)
                #print()

                y_true_total.extend(y_true)
                y_pred_binary_total.extend(y_pred_binary)
                # y_pred_probab_total.extend(y_pred_probab)
                if batch_iteration[phase] % len(dloader) != 1:
                    y_pred_logits_total = torch.cat((y_pred_logits_total, pred_logits))
                else:
                    y_pred_logits_total = pred_logits

                # losses
                batch_loss = loss.item() * x.shape[0]  # loss of whole batch (loss*batchsize (as loss was averaged ('mean')), each item of batch had this loss on avg)
                running_loss += batch_loss

                """
                # array([[a, b],
                #        [c, d]])
                # a: tn (0 true, 0 predicted)  # negative is predicted, and the prediction is true
                # b: fp (0 true, 1 predicted)  # positive is predicted, and the prediction is wrong
                # c: fn (1 true, 0 predicted)  # negative is predicted, and the prediction is wrong
                # d: tp (1 true, 1 predicted)  # positive is predicted, and the prediction is true
                # to extract all values:
                # (tn, fp, fn, tp) = cm.ravel()
                """

                if phase == 'train':
                    if (batch_iteration[phase]%LOG_EVERY) == 0:
                        loss = running_loss/LOG_EVERY
                        print(f'batch iteration: {batch_iteration[phase]} / {len(dloader)*(epoch+1)} with {phase} loss (avg over {LOG_EVERY} batch iterations): {loss}')

                        get_and_log_metrics(yt=y_true_total, ypred=y_pred_binary_total, ylogits=y_pred_logits_total, ep=epoch, batch_it_loss=loss, ph=phase, bi=batch_iteration[phase])

                        running_loss = 0

                if phase == 'val':
                    if batch_iteration[phase]%len(dloader) == 0:
                        loss = running_loss/len(dloader)
                        print(f'batch iteration: {batch_iteration[phase]} / {len(dloader)*(epoch+1)} ... {phase} loss (avg over whole validation dataloader): {loss}')

                        get_and_log_metrics(yt=y_true_total, ypred=y_pred_binary_total, ylogits=y_pred_probab_total, ep=epoch, batch_it_loss=loss, ph=phase, bi=batch_iteration[phase])
                        # as we're in last loop for validation, running_loss will be set to 0 anyways (changing the phase back to train)

            if phase == 'train':  # at end of epoch (training, could also be end of validation)
                if scheduler is not None:
                    scheduler.step()

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
parser.add_argument('--model', help='choose model type')

args = parser.parse_args()

LOGGING = True
if LOGGING:
    wandb.init(project="model_fog_classification", entity="jbaumer", config=args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # set device


############ GLOBAL VARIABLES ############

BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
EPOCHS = args.epochs
TRAIN_SPLIT = args.train_split
WEIGHTED = args.weighted
PATH_DATASET = args.path_dset
LR_SCHEDULER = args.lr_scheduler
MODEL_TYPE = args.model

STATIONS_CAM_STR = args.stations_cam
STATIONS_CAM_STR = STATIONS_CAM_STR.replace("\\", "")
STATIONS_CAM_LST = sorted(ast.literal_eval(STATIONS_CAM_STR))  # sort to make sure not two models with data from same cameras (but input in different order) will be saved

N_CLASSES = 2
PATH_MODEL = f'models/{STATIONS_CAM_LST}_bs_{BATCH_SIZE}_LR_{LEARNING_RATE}_epochs_{EPOCHS}_weighted_{WEIGHTED}_lr_sched_{LR_SCHEDULER}'
LOG_EVERY = 15
LOAD_MODEL = False


############ DATASETS AND DATALOADERS ############

# new approach: use only handlabeled data for dset full, then sep. into train and val
#dset_full = DischmaSet_classification(root=PATH_DATASET, stat_cam_lst=STATIONS_CAM_LST, mode='full_v2')
#dset_train, dset_val = get_train_val_split(dset_full)
"""
# alternative, in case train and val sets should be got from different sources (manual labels vs labels from txt files - and different dates)
dset_train = DischmaSet_classification(root=PATH_DATASET, stat_cam_lst=STATIONS_CAM_LST, mode='train')
dset_val = DischmaSet_classification(root=PATH_DATASET, stat_cam_lst=STATIONS_CAM_LST, mode='val')
"""
dset_train = DischmaSet_classification(root=PATH_DATASET, stat_cam_lst=STATIONS_CAM_LST, mode='train')
dset_val = DischmaSet_classification(root=PATH_DATASET, stat_cam_lst=STATIONS_CAM_LST, mode='val')
dset_test = DischmaSet_classification(root=PATH_DATASET, stat_cam_lst=STATIONS_CAM_LST, mode='test')

print(f'Dischma sets (train, val and test) with data from {STATIONS_CAM_LST} created.')

dloader_train = DataLoader(dataset=dset_train, batch_size=BATCH_SIZE, shuffle=True)
dloader_val = DataLoader(dataset=dset_val, batch_size=BATCH_SIZE)
dloader_test = DataLoader(dataset=dset_test, batch_size=BATCH_SIZE)

# Note:
#   class 0: not foggy
#   class 1: foggy


############ MODEL, LOSS, OPTIMIZER, SCHEDULER ############

if WEIGHTED == 'False':
    weights = None
elif WEIGHTED == 'Manual':
    weights = torch.Tensor([0.2, 0.8]).to(device)  # w0 smaller, w1 larger because we want a high recall (only few FN) - when we predict a negative, we must be sure that it is negative (sunny)
elif WEIGHTED == 'Auto':  # TODO: check if works (with new dataset class)
    n_class_0, n_class_1 = dset_train.get_balancedness()  # balancedness from full dataset, not only from train - but should have similar distribution
    n_tot = n_class_0 + n_class_1
    w0, w1 = n_class_1/n_tot, n_class_0/n_tot
    weights = torch.Tensor([w0, w1]).to(device)

if MODEL_TYPE == 'resnet':
    # should work
    model = models.resnet18(pretrained=True)
    n_features = model.fc.in_features  # adapt fully connected layer
    model.fc = nn.Linear(n_features, N_CLASSES)  # note: Softmax (from real to probab) is implicitly applied when working with crossentropyloss

elif MODEL_TYPE == 'efficientnet':
    # too much memory used
    model = models.efficientnet_b1(pretrained=True)
    n_features = model.classifier[1].in_features  # adapt fully connected layer
    model.classifier[1] = nn.Linear(n_features, N_CLASSES)

elif MODEL_TYPE == 'densenet':
    # too much memory used
    model = models.densenet121()
    n_features = model.classifier.in_features
    model.classifier = nn.Linear(n_features, N_CLASSES)

elif MODEL_TYPE == 'vgg11':
    # should work
    model = models.vgg11()
    f_in = model.classifier[6].out_features
    model.classifier.add_module('new_relu', torch.nn.modules.activation.ReLU(inplace=True))  # add to append to last sequential block (of the classifier)
    model.classifier.add_module('new_dropout', torch.nn.modules.dropout.Dropout(p=0.5, inplace=False))
    model.classifier.add_module('new_linear', torch.nn.modules.Linear(in_features=f_in, out_features=N_CLASSES))

elif MODEL_TYPE == 'googlenet':
    model = models.GoogLeNet(num_classes=N_CLASSES)

if LOAD_MODEL:
    if os.path.exists(PATH_MODEL) == True:
        print('trained model already exists, loading model...')
        model = torch.load(PATH_MODEL)

model = model.to(device)

# note: Softmax (from real to probab) is implicitly applied when working with crossentropyloss
criterion = nn.CrossEntropyLoss(reduction='mean', weight=weights)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)
if LR_SCHEDULER != 'None':
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=int(LR_SCHEDULER), gamma=0.5)  # Decay LR by a factor of gamma every 'step_size' epochs
elif LR_SCHEDULER == 'None':
    exp_lr_scheduler = None
print('criterion: ', criterion, 'optimizer: ', optimizer, 'lr scheduler: ', exp_lr_scheduler)

train_val_model(model=model, criterion=criterion, optimizer=optimizer, scheduler=exp_lr_scheduler, num_epochs=EPOCHS)

test_model(model=model)