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
    # assumption: learned features from resnet (resp. other models) can also be used for our images (same pixel distribution)
        # (might be slightly wrong, as we are using imgs with more snow/fog than normal (imagenet) - mean and std will be slightly off (not 0, resp 1))
    # train/validate/test model - evaluations only with handlabelled data


################# FUNCTIONS ######################

def getmetrics_matchPrec(ytrue, yprob_pos):
    """
    get metrics by choosing the threshold (from pr-curve),
    so that baseline precision is achieved
    interesting to see how much recall can be achieved
    """
    prec_vec, rec_vec, th_vec = precision_recall_curve(y_true=ytrue, probas_pred=yprob_pos)
    F1s_vec =  (1 + BETA**2)* (prec_vec*rec_vec) / (BETA**2 * prec_vec + rec_vec + EPSILON)  # same shape as prec_vec and rec_vec
    best_ind = find_nearest(array=prec_vec, value=PRECISION_BASELINE)
    opt_f1 = F1s_vec[best_ind]  # optimal f1-score
    opt_prec, opt_rec = prec_vec[best_ind], rec_vec[best_ind]
    opt_thresh = th_vec[best_ind]

    return opt_prec, opt_rec, opt_f1, opt_thresh  # all scalar values


def getmetrics_matchRec(ytrue, yprob_pos):
    """
    get metrics by choosing the threshold (from pr-curve),
    so that baseline recall is achieved
    interesting to see how much precision can be achieved
    """
    prec_vec, rec_vec, th_vec = precision_recall_curve(y_true=ytrue, probas_pred=yprob_pos)
    F1s_vec =  (1 + BETA**2)* (prec_vec*rec_vec) / (BETA**2 * prec_vec + rec_vec + EPSILON)  # same shape as prec_vec and rec_vec
    best_ind = find_nearest(array=rec_vec, value=RECALL_BASELINE)
    opt_f1 = F1s_vec[best_ind]  # optimal f1-score
    opt_prec, opt_rec = prec_vec[best_ind], rec_vec[best_ind]
    opt_thresh = th_vec[best_ind]

    return opt_prec, opt_rec, opt_f1, opt_thresh  # all scalar values


def find_nearest(array, value):
    """
    return index of that val in the %array that is closest to %value
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def get_optimal_prec_rec_f1_th_and_prtable(ytrue, yprob_pos):
    """
    get metrics by choosing the threshold (from pr-curve),
    so that optimal f1-score is achieved
    interesting to see how much better f1-score can be achieved

    note:
    for the test metrics, the threshold which yielded the optimal f1-score in the last validation loop is used.
    """
    prec_vec, rec_vec, th_vec = precision_recall_curve(y_true=ytrue, probas_pred=yprob_pos)
    F1s_vec =  (1 + BETA**2)*(prec_vec*rec_vec) / (BETA**2 * prec_vec + rec_vec + EPSILON)  # same shape as prec_vec and rec_vec
    best_ind = np.argmax(F1s_vec) 
    opt_f1 = F1s_vec[best_ind]  # optimal f1-score
    opt_prec, opt_rec = prec_vec[best_ind], rec_vec[best_ind]
    opt_thresh = th_vec[best_ind]
    table = wandb.Table(data=[[x, y] for (x, y) in zip(prec_vec, rec_vec)], columns = ["Precision", "Recall"])
    # table = wandb.Table(data=[[x, y] for (x, y) in zip(precs[::len(precs)//9800], recs[::len(recs)//9800])], columns = ["Precision", "Recall"]) 

    return opt_prec, opt_rec, opt_f1, opt_thresh, table  # all scalar values (except for table)


def get_and_log_metrics(yt, ypred_th_std, ylogits, ep, batch_it_loss, ph, bi=0, ypred_th_optimal=None):
    """
    yt, ypred_th_std: lists
    yprob: torch tensor
    """
    yprobab = torch.softmax(ylogits, dim=1)  # [log_every*batchsize, 2] probas between 0 and 1
    yprobab_neg = yprobab[:, 0]
    yprobab_pos = yprobab[:, 1]  # compute precision and recall for class one

    acc_std = accuracy_score(y_true=yt, y_pred=ypred_th_std)
    prec_std = precision_score(y_true=yt, y_pred=ypred_th_std)
    rec_std = recall_score(y_true=yt, y_pred=ypred_th_std)
    f1_std = f1_score(y_true=yt, y_pred=ypred_th_std)

    if LOGGING:
        print(f'logging the {ph} set metrics...')

        if ph == 'train':
            wandb.log({
                'n_epoch' : ep,
                'batch_iteration' : bi,
                f'{ph}/loss' : batch_it_loss,

                f'{ph}/accuracy_th_std' : acc_std,
                f'{ph}/precision_th_std' : prec_std,
                f'{ph}/recall_th_std' : rec_std,  # this should be high !!! (to catch all foggy images)
                f'{ph}/F1-score_th_std' : f1_std,
                f'{ph}/conf_mat_th_std' : wandb.plot.confusion_matrix(y_true=yt, preds=ypred_th_std, class_names=['class 0 (not foggy)', 'class 1 (foggy)']),
                #f'{ph}/precision_recall_curve' : wandb.plot.pr_curve(y_true=yt, y_probas=yprobab.detach().cpu(), labels=['class 0 (not foggy)', 'class 1 (foggy)']),
                })
        
        if ph == 'val':
            # also log p-r-curve
            # also log optimal metrics
            # also get optimal threshold

            # get optimal metrics
            opt_prec, opt_rec, opt_f1, opt_thresh, prtable = get_optimal_prec_rec_f1_th_and_prtable(ytrue=yt, yprob_pos=yprobab_pos.cpu().detach())
            prec_MatchPrec, rec_MatchPrec, f1_MatchPrec, thresh_MatchPrec = getmetrics_matchPrec(ytrue=yt, yprob_pos=yprobab_pos.cpu().detach())
            prec_MatchRec, rec_MatchRec, f1_MatchRec, thresh_MatchRec = getmetrics_matchRec(ytrue=yt, yprob_pos=yprobab_pos.cpu().detach())

            global OPTIMAL_THRESHOLD  # where best f1-score is achieved
            OPTIMAL_THRESHOLD = opt_thresh  # after last loop, this variable can be taken for test set threshold (as other fct-call, make variable global), is used is test-function
            global THRESHOLD_MATCHPREC
            THRESHOLD_MATCHPREC = thresh_MatchPrec
            global THRESHOLD_MATCHREC
            THRESHOLD_MATCHREC = thresh_MatchRec

            wandb.log({
            'n_epoch' : ep,
            'batch_iteration' : bi,
            f'{ph}/loss' : batch_it_loss,
            f'{ph}/PR_Curve' : wandb.plot.line(prtable, 'Precision', 'Recall', title='PR-Curve'),

            f'{ph}/threshold_standard/accuracy' : acc_std,
            f'{ph}/threshold_standard/precision' : prec_std,
            f'{ph}/threshold_standard/recall' : rec_std,  # this should be high !!! (to catch all foggy images)
            f'{ph}/threshold_standard/F1-score' : f1_std,
            f'{ph}/threshold_standard/conf_mat' : wandb.plot.confusion_matrix(y_true=yt, preds=ypred_th_std, class_names=['class 0 (not foggy)', 'class 1 (foggy)']),

            f'{ph}/threshold_optimal/precision' : opt_prec,
            f'{ph}/threshold_optimal/recall' : opt_rec,
            f'{ph}/threshold_optimal/f1-score' : opt_f1,
            f'{ph}/threshold_optimal/threshold_opt' : opt_thresh,

            f'{ph}/threshold_MatchPrec/precision' : prec_MatchPrec,
            f'{ph}/threshold_MatchPrec/recall' : rec_MatchPrec,
            f'{ph}/threshold_MatchPrec/f1-score' : f1_MatchPrec,
            f'{ph}/threshold_MatchPrec/threshold_opt' : thresh_MatchPrec,

            f'{ph}/threshold_MatchRec/precision' : prec_MatchRec,
            f'{ph}/threshold_MatchRec/recall' : rec_MatchRec,
            f'{ph}/threshold_MatchRec/f1-score' : f1_MatchRec,
            f'{ph}/threshold_MatchRec/threshold_opt' : thresh_MatchRec,

            })

        if ph == 'test':
            # for predictions with optimal threshold: using optimal threshold from last validation loop

            # get predictions with th matchPrecision:
            prec_MatchPrec, rec_MatchPrec, f1_MatchPrec, thresh_MatchPrec = getmetrics_matchPrec(ytrue=yt, yprob_pos=yprobab_pos.cpu().detach())

            # get predictions with th matchRecall:
            prec_MatchRec, rec_MatchRec, f1_MatchRec, thresh_MatchRec = getmetrics_matchRec(ytrue=yt, yprob_pos=yprobab_pos.cpu().detach())
            
            # if th_opt is got from last validation loop
            th_opt = torch.tensor([OPTIMAL_THRESHOLD]).to(device)
            pred_binary_th_optimal = (yprobab_pos > th_opt).float().cpu().tolist()  # threshold: last from validation # either 0 or 1 / shape: batchsize (8)

            acc_optimal = accuracy_score(y_true=yt, y_pred=pred_binary_th_optimal)
            prec_optimal = precision_score(y_true=yt, y_pred=pred_binary_th_optimal)
            rec_optimal = recall_score(y_true=yt, y_pred=pred_binary_th_optimal)
            f1_optimal = f1_score(y_true=yt, y_pred=pred_binary_th_optimal)

            wandb.log({
            'n_epoch' : ep,
            'batch_iteration' : bi,
            f'{ph}/loss' : batch_it_loss,
            # f'{ph}/PR_Curve' : wandb.plot.line(prtable, 'Precision', 'Recall', title='PR-Curve'),

            f'{ph}/threshold_standard/accuracy' : acc_std,
            f'{ph}/threshold_standard/precision' : prec_std,
            f'{ph}/threshold_standard/recall' : rec_std,  # this should be high !!! (to catch all foggy images)
            f'{ph}/threshold_standard/F1-score' : f1_std,
            f'{ph}/threshold_standard/conf_mat' : wandb.plot.confusion_matrix(y_true=yt, preds=ypred_th_std, class_names=['class 0 (not foggy)', 'class 1 (foggy)']),
            
            f'{ph}/threshold_optimal/accuracy' : acc_optimal,
            f'{ph}/threshold_optimal/precision' : prec_optimal,
            f'{ph}/threshold_optimal/recall' : rec_optimal,
            f'{ph}/threshold_optimal/f1-score' : f1_optimal,
            f'{ph}/threshold_optimal/threshold_opt' : OPTIMAL_THRESHOLD,

            f'{ph}/threshold_MatchPrec/precision' : prec_MatchPrec,
            f'{ph}/threshold_MatchPrec/recall' : rec_MatchPrec,
            f'{ph}/threshold_MatchPrec/f1-score' : f1_MatchPrec,
            f'{ph}/threshold_MatchPrec/threshold_opt' : thresh_MatchPrec,

            f'{ph}/threshold_MatchRec/precision' : prec_MatchRec,
            f'{ph}/threshold_MatchRec/recall' : rec_MatchRec,
            f'{ph}/threshold_MatchRec/f1-score' : f1_MatchRec,
            f'{ph}/threshold_MatchRec/threshold_opt' : thresh_MatchRec,

            })

        print('logging complete.')


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
    y_reshaped = y.reshape(2, -1).numpy()
    grid_img = torchvision.utils.make_grid(x, nrow=int(batchsize/2), normalize=True)
    plt.title(f'batch iteration: {bi}\n{y_reshaped[0]}\n{y_reshaped[1]}')
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.savefig(f'stats/fig_check_manually/grid_batch_iteration_{bi}')

def append_misclassified(x, yt, ypred_bin):
    global x_misclassified
    for y_idx in range(len(yt)):
        if yt[y_idx] != ypred_bin[y_idx]:
            current_x = x[y_idx]
            current_x = current_x.unsqueeze(0)
            x_misclassified = torch.cat((x_misclassified, current_x))
            print()

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
    y_true_total = []
    y_pred_logits_total = None
    y_pred_binary_total_th_std = []
    y_pred_binary_total_th_optimal = []

    global x_misclassified
    x_misclassified = torch.Tensor().to(device)

    for x, y in dloader:
        batch_iteration[phase] += 1

        # move to GPU
        x = x.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(phase == 'train'):
            pred_logits = model(x)  # predictions (logits) (for class 0 and 1) / shape: batchsize, nclasses (8,2)
            loss = criterion(pred_logits, y)

            yprobab = torch.softmax(pred_logits, dim=1)  # [log_every*batchsize, 2] probas between 0 and 1, sum up to one
            yprobab_pos = yprobab[:, 1]  # compute metrics for class one  (yprobab_neg = yprobab[:, 0])          

            if not  LOAD_MODEL:
                threshold = torch.tensor([OPTIMAL_THRESHOLD]).to(device)

            pred_binary_th_std = yprobab.argmax(dim=1)  # threshold 0.5 # either 0 or 1 / shape: batchsize (8) / takes higher probablity (from the two classes) to compare to y (y_true)
            if not  LOAD_MODEL:
                pred_binary_th_optimal = (yprobab_pos > threshold).float()  # threshold: last from validation # either 0 or 1 / shape: batchsize (8)

        # stats
        y_true = y.cpu().tolist()
        y_pred_binary_th_std = pred_binary_th_std.cpu().tolist()
        if not  LOAD_MODEL:
            y_pred_binary_th_optimal = pred_binary_th_optimal.cpu().tolist()
        
        append_misclassified(x, y_true, y_pred_binary_th_std)


        y_true_total.extend(y_true)
        y_pred_binary_total_th_std.extend(y_pred_binary_th_std)
        if not  LOAD_MODEL:
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
            get_and_log_metrics(yt=y_true_total, ypred_th_std=y_pred_binary_total_th_std, ylogits=y_pred_logits_total, ep=epoch, batch_it_loss=loss, ph=phase, bi=batch_iteration[phase], ypred_th_optimal=y_pred_binary_total_th_optimal)

    time_end = time()
    time_elapsed = time_end - time_start
    print(f'testing (on {device}) completed in {time_elapsed} seconds.')


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
            y_pred_logits_total = None
            y_pred_binary_total = []

            for x, y in dloader:
                #for i in range(1000):  # to check if one ele can be learned
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
                    pred_binary = pred_logits.argmax(dim=1)  # [8], threshold 0.5 # either 0 or 1 / shape: batchsize (8) / take higher value (from the two classes) to compare to y (y_true)

                    loss = criterion(pred_logits, y)
                    if phase == 'train':
                        loss.backward()  # backprop
                        optimizer.step()  # update params

                # stats
                y_true = y.cpu().tolist()

                y_pred_binary = pred_binary.cpu().tolist() 

                # debug - e.g. if problematic that y_pred_binary is full with 0 (1 never gets predicted)
                #print(y_true)
                #print(y_pred_binary)
                #print()

                # accumulate GT and binary predictions (with std threshold)
                y_true_total.extend(y_true)
                y_pred_binary_total.extend(y_pred_binary)

                # accumulate logits (conversion to probab happens later) to log metrics with different thresholds
                if batch_iteration[phase] % len(dloader) != 1:
                    y_pred_logits_total = torch.cat((y_pred_logits_total, pred_logits))
                else:
                    y_pred_logits_total = pred_logits

                # losses
                batch_loss = loss.item() * x.shape[0]  # loss of whole batch (loss*batchsize (as loss was averaged ('mean')), each item of batch had this loss on avg)
                running_loss += batch_loss

                if phase == 'train':
                    if (batch_iteration[phase]%LOG_EVERY) == 0:
                        loss = running_loss/LOG_EVERY
                        print(f'batch iteration: {batch_iteration[phase]} / {len(dloader)*(epoch+1)} with {phase} loss (avg over {LOG_EVERY} batch iterations): {loss}')
                        get_and_log_metrics(yt=y_true_total, ypred_th_std=y_pred_binary_total, ylogits=y_pred_logits_total, ep=epoch, batch_it_loss=loss, ph=phase, bi=batch_iteration[phase])
                        running_loss = 0

                if phase == 'val':
                    if batch_iteration[phase]%len(dloader) == 0:  # last validation loop
                        loss = running_loss/len(dloader)
                        print(f'batch iteration: {batch_iteration[phase]} / {len(dloader)*(epoch+1)} ... {phase} loss (avg over whole validation dataloader): {loss}')
                        get_and_log_metrics(yt=y_true_total, ypred_th_std=y_pred_binary_total, ylogits=y_pred_logits_total, ep=epoch, batch_it_loss=loss, ph=phase, bi=batch_iteration[phase])
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
parser.add_argument('--test_on_dset', help='if None, test on same set as model was trained - this variable is to make sure a model can be tested on single cam if trained on all cams')
parser.add_argument('--weighted', help='how to weight the classes (manual: as given in script / Auto: Inversely proportional to occurance / False: not at all')
parser.add_argument('--path_dset', help='path to used dataset ')
parser.add_argument('--lr_scheduler', help='whether to use a lr scheduler, and if so after how many epochs to reduced LR')
parser.add_argument('--model', help='choose model type')
parser.add_argument('--optim', help='set type of optimizer (Adam or SGD)')
parser.add_argument('--weight_decay', type=float, help='set weight decay (used for Adam and for SGD')
parser.add_argument('--momentum', type=float, help='set momentum used for SGD optimizer')

args = parser.parse_args()

LOGGING = False
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
OPTIM = args.optim
WEIGHT_DECAY = args.weight_decay
MOMENTUM = args.momentum

STATIONS_CAM_STR = args.stations_cam
STATIONS_CAM_STR = STATIONS_CAM_STR.replace("\\", "")
STATIONS_CAM_LST = sorted(ast.literal_eval(STATIONS_CAM_STR))  # sort to make sure not two models with data from same cameras (but input in different order) will be saved

TEST_ON_DSET_STR = args.stations_cam
TEST_ON_DSET_STR = TEST_ON_DSET_STR.replace("\\", "")
TEST_ON_DSET_LST = ast.literal_eval(TEST_ON_DSET_STR)  # sort to make sure not two models with data from same cameras (but input in different order) will be saved


N_CLASSES = 2
BETA = 1
EPSILON = 0
PATH_MODEL = f'models/{STATIONS_CAM_LST}_bs_{BATCH_SIZE}_LR_{LEARNING_RATE}_epochs_{EPOCHS}_weighted_{WEIGHTED}_lr_sched_{LR_SCHEDULER}'
PATH_LOAD_MODEL = f'final_models_classification_v01/{STATIONS_CAM_LST}_bs_{BATCH_SIZE}_LR_{LEARNING_RATE}_epochs_{EPOCHS}_weighted_{WEIGHTED}_lr_sched_{LR_SCHEDULER}'

LOG_EVERY = 200
LOAD_MODEL = True

BASELINES = {
    "['Buelenberg_1']" : {'PRECISION': 0.52, 'RECALL': 0.913},
    "['Buelenberg_2']":{'PRECISION': 0.472, 'RECALL': 0.862},
    "['Giementaelli_1']":{'PRECISION': 0.500, 'RECALL': 0.653},
    "['Giementaelli_2']":{'PRECISION': 0.299, 'RECALL': 0.442},
    "['Giementaelli_3']":{'PRECISION': 0.444, 'RECALL': 0.453},
    "['Luksch_1']":{'PRECISION': 0.31, 'RECALL': 0.926},
    "['Luksch_2']":{'PRECISION': 0.481, 'RECALL': 0.633},
    "['Sattel_1']":{'PRECISION': 0.328, 'RECALL': 0.955},
    "['Sattel_2']":{'PRECISION': 0.267, 'RECALL': 0.917},
    "['Sattel_3']":{'PRECISION': 0.253, 'RECALL': 0.952},
    "['Stillberg_1']":{'PRECISION': 0.428, 'RECALL': 0.812},
    "['Stillberg_2']":{'PRECISION': 0.44, 'RECALL': 0.827},
    "['Stillberg_3']":{'PRECISION': 0.393, 'RECALL': 0.932},
    "['Buelenberg_1', 'Buelenberg_2', 'Giementaelli_1', 'Giementaelli_2', 'Giementaelli_3', 'Luksch_1', 'Luksch_2', 'Sattel_1', 'Sattel_2', 'Sattel_3', 'Stillberg_1', 'Stillberg_2', 'Stillberg_3']" : {'PRECISION': 0.398, 'RECALL': 0.83}
}
PRECISION_BASELINE = BASELINES[str(STATIONS_CAM_LST)]['PRECISION']
RECALL_BASELINE = BASELINES[str(STATIONS_CAM_LST)]['RECALL']

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

if not TEST_ON_DSET_LST: # no special list given to evaluate on
    dset_test = DischmaSet_classification(root=PATH_DATASET, stat_cam_lst=STATIONS_CAM_LST, mode='test')
else:  # a list is given to eval test set on
    dset_test = DischmaSet_classification(root=PATH_DATASET, stat_cam_lst=TEST_ON_DSET_LST, mode='test')


print(f'Dischma sets (train, val and test) with data from {STATIONS_CAM_LST} created.')

dloader_train = DataLoader(dataset=dset_train, batch_size=BATCH_SIZE, shuffle=True)
dloader_val = DataLoader(dataset=dset_val, batch_size=BATCH_SIZE)
dloader_test = DataLoader(dataset=dset_test, batch_size=BATCH_SIZE)
print('lengths (train, val, test dloader): ', len(dloader_train), len(dloader_val), len(dloader_test))
print('distribution of neg, pos labels: ')
print(f'for train set: {dset_train.get_balancedness()}')
print(f'for val set: {dset_val.get_balancedness()}')
print(f'for test set: {dset_test.get_balancedness()}')

# Note:
#   class 0: not foggy
#   class 1: foggy


############ MODEL, LOSS, OPTIMIZER, SCHEDULER ############

if WEIGHTED == 'False':
    weights = None
elif WEIGHTED == 'Manual':
    weights = torch.Tensor([0.2, 0.8]).to(device)  # w0 smaller, w1 larger because we want a high recall (only few FN) - when we predict a negative, we must be sure that it is negative (sunny)
elif WEIGHTED == 'Auto':
    n_class_0, n_class_1 = dset_train.get_balancedness()  # balancedness from full dataset, not only from train - but should have similar distribution
    n_tot = n_class_0 + n_class_1
    w0, w1 = n_class_1/n_tot, n_class_0/n_tot
    weights = torch.Tensor([w0, w1]).to(device)

if MODEL_TYPE == 'resnet':
    # should work
    model = models.resnet50(pretrained=True)
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
    if os.path.exists(PATH_LOAD_MODEL) == True:
        print('trained model already exists, loading model...')
        model = torch.load(PATH_LOAD_MODEL)

model = model.to(device)

# note: Softmax (from real to probab) is implicitly applied when working with crossentropyloss
criterion = nn.CrossEntropyLoss(reduction='mean', weight=weights)
if OPTIM == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
elif OPTIM == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)  # weight_decay: It is used for adding the l2 penality to the loss (default = 0)

if LR_SCHEDULER != 'None':
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=int(LR_SCHEDULER), gamma=0.8)  # Decay LR by a factor of gamma every 'step_size' epochs
elif LR_SCHEDULER == 'None':
    exp_lr_scheduler = None
print('criterion: ', criterion, 'optimizer: ', optimizer, 'lr scheduler: ', exp_lr_scheduler)

if not LOAD_MODEL:
    train_val_model(model=model, criterion=criterion, optimizer=optimizer, scheduler=exp_lr_scheduler, num_epochs=EPOCHS)

test_model(model=model)


def plot_misclassifications():

    plt.title(f'misclassifications from {STATIONS_CAM_LST}')
    grid_img = torchvision.utils.make_grid(x_misclassified.cpu(), normalize=True)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.savefig(f'misclassifications/misclassification_cams_{STATIONS_CAM_LST}')

plot_misclassifications()

