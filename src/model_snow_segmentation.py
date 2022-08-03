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
from torchvision import transforms


from dischma_set_segmentation import DischmaSet_segmentation

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

import warnings
import rasterio
from tqdm import tqdm

import matplotlib

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('imports done')

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class SimpleNetwork(torch.nn.Module):
    """Mini segmentation network."""

    def __init__(self, in_channels, classes):
        super(SimpleNetwork, self).__init__()

        self.in_channels = in_channels
        self.classes = classes

        self.convs = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, classes, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.convs(x)


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

def get_optimal_prec_rec_f1_th_and_prtable(ytrue, yprob_pos):
    """
    get metrics by choosing the threshold (from pr-curve),
    so that optimal f1-score is achieved
    interesting to see how much better f1-score can be achieved

    note:
    for the test metrics, the threshold which yielded the optimal f1-score in the last validation loop is used.
    """
    prec_vec, rec_vec, th_vec = precision_recall_curve(y_true=ytrue, probas_pred=yprob_pos)
    F1s_vec = (1 + BETA**2)*(prec_vec*rec_vec) / (BETA**2 * prec_vec + rec_vec + EPSILON)  # same shape as prec_vec and rec_vec
    best_ind = np.nanargmax(F1s_vec)  # nanargmax ignores all NaN values
    opt_f1 = F1s_vec[best_ind]  # optimal f1-score
    opt_prec, opt_rec = prec_vec[best_ind], rec_vec[best_ind]
    opt_thresh = th_vec[best_ind]
    table = wandb.Table(data=[[x, y] for (x, y) in zip(prec_vec, rec_vec)], columns = ["Precision", "Recall"])
    # table = wandb.Table(data=[[x, y] for (x, y) in zip(precs[::len(precs)//9800], recs[::len(recs)//9800])], columns = ["Precision", "Recall"]) 

    return opt_prec, opt_rec, opt_f1, opt_thresh, table  # all scalar values (except for table)

def get_and_log_metrics(yt, yt_short, ypred_bin, ypred_logits, ep, batch_it_loss, ph, bi):
    """
    yt: y_true_stack_total
    yt_short: y_true_stack_selection
    ypred_bin: y_pred_binary_stack_total - with standard threshold
    ypred_logits: y_pred_logits_nosnow_stack_selection
    
    yt and ypred_bin should have same lenght
    yt_short and ypred_logits should have same length
    """

    y_probab_pos = torch.sigmoid(ypred_logits)  # probabilities for class one (nosnow), because given like this as input to function

    #yt = yt.tolist()
    #yt_short = yt_short.tolist()
    #ypred_bin = ypred_bin.tolist()
    #ypred_logits = ypred_logits.tolist()

    acc_std = accuracy_score(y_true=yt, y_pred=ypred_bin)
    prec_std = precision_score(y_true=yt, y_pred=ypred_bin)
    rec_std = recall_score(y_true=yt, y_pred=ypred_bin)
    f1_std = f1_score(y_true=yt, y_pred=ypred_bin)

    if LOGGING:
        print(f'logging the {ph} set metrics...')

        if ph == 'train':
            wandb.log({
                'n_epoch' : ep,
                'batch_iteration' : bi,
                f'{ph}/loss' : batch_it_loss,

                f'{ph}/accuracy_th_std' : acc_std,
                f'{ph}/precision_th_std' : prec_std,
                f'{ph}/recall_th_std' : rec_std,
                f'{ph}/F1-score_th_std' : f1_std,  # this should be high for good segmentation
                f'{ph}/conf_mat_th_std' : wandb.plot.confusion_matrix(y_true=yt.tolist(), preds=ypred_bin.tolist(), class_names=['class 0 (snow)', 'class 1 (no snow)']),
                #f'{ph}/precision_recall_curve' : wandb.plot.pr_curve(y_true=yt, y_probas=yprobab.detach().cpu(), labels=['class 0 (not foggy)', 'class 1 (foggy)']),
                })

        if ph == 'val':
            # also log p-r-curve
            # also log optimal metrics
            # also get optimal threshold

            # get optimal metrics
            # here, only every 1000th element is taken (for yt_short and yprob_pob)
            opt_prec, opt_rec, opt_f1, opt_thresh, prtable = get_optimal_prec_rec_f1_th_and_prtable(ytrue=yt_short.cpu().detach(), yprob_pos=y_probab_pos.cpu().detach())

            global OPTIMAL_THRESHOLD  # where best f1-score is achieved
            OPTIMAL_THRESHOLD = opt_thresh  # after last loop, this variable can be taken for test set threshold (as other fct-call, make variable global), is used is test-function

            wandb.log({
            'n_epoch' : ep,
            'batch_iteration' : bi,
            f'{ph}/loss' : batch_it_loss,
            f'{ph}/PR_Curve' : wandb.plot.line(prtable, 'Precision', 'Recall', title='PR-Curve'),

            f'{ph}/threshold_standard/accuracy' : acc_std,
            f'{ph}/threshold_standard/precision' : prec_std,
            f'{ph}/threshold_standard/recall' : rec_std,  # this should be high !!! (to catch all foggy images)
            f'{ph}/threshold_standard/F1-score' : f1_std,
            f'{ph}/threshold_standard/conf_mat' : wandb.plot.confusion_matrix(y_true=yt.tolist(), preds=ypred_bin.tolist(), class_names=['class 0 (snow)', 'class 1 (no snow)']),

            f'{ph}/threshold_optimal/precision' : opt_prec,
            f'{ph}/threshold_optimal/recall' : opt_rec,
            f'{ph}/threshold_optimal/f1-score' : opt_f1,
            f'{ph}/threshold_optimal/threshold_opt' : opt_thresh,

            })


        if ph == 'test':
            # TODO: do sth similar to what is done in model_fog_classification

            # for predictions with optimal threshold: using optimal threshold from last validation loop
            # if th_opt is got from last validation loop
            th_opt = torch.tensor([OPTIMAL_THRESHOLD])

            ### TODO check from here on !!!
            pred_binary_th_optimal = (y_probab_pos > th_opt).float().cpu().tolist()  # threshold: last from validation # either 0 or 1 / shape: ???

            acc_optimal = accuracy_score(y_true=yt_short, y_pred=pred_binary_th_optimal)
            prec_optimal = precision_score(y_true=yt_short, y_pred=pred_binary_th_optimal)
            rec_optimal = recall_score(y_true=yt_short, y_pred=pred_binary_th_optimal)
            f1_optimal = f1_score(y_true=yt_short, y_pred=pred_binary_th_optimal)

            wandb.log({
            'n_epoch' : ep,
            'batch_iteration' : bi,
            f'{ph}/loss' : batch_it_loss,
            # f'{ph}/PR_Curve' : wandb.plot.line(prtable, 'Precision', 'Recall', title='PR-Curve'),

            f'{ph}/threshold_standard/accuracy' : acc_std,
            f'{ph}/threshold_standard/precision' : prec_std,
            f'{ph}/threshold_standard/recall' : rec_std,  # this should be high !!! (to catch all foggy images)
            f'{ph}/threshold_standard/F1-score' : f1_std,
            f'{ph}/threshold_standard/conf_mat' : wandb.plot.confusion_matrix(y_true=yt.tolist(), preds=ypred_bin.tolist(), class_names=['class 0 (not foggy)', 'class 1 (foggy)']),
            
            f'{ph}/threshold_optimal/accuracy' : acc_optimal,
            f'{ph}/threshold_optimal/precision' : prec_optimal,
            f'{ph}/threshold_optimal/recall' : rec_optimal,
            f'{ph}/threshold_optimal/f1-score' : f1_optimal,
            f'{ph}/threshold_optimal/threshold_opt' : OPTIMAL_THRESHOLD,

            })


'''def get_and_log_metrics(yt, ypred, ep, batch_it_loss, ph, bi=0):
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
'''

def test_model(model):
    # with trained model, predict class for every x,y in test set
    # log the respective metrics (only one value per metric)
    # TODO: plot the incorrect segmentations

    time_start = time()
    epoch = 0
    batch_iteration = {}
    batch_iteration['test'] = 0

    phase = 'test'
    print(f'{phase} phase starting...')
    model.eval()
    dloader = dloader_test
    #TODO: correct dloader_test -> DONE
    running_loss = 0  # loss (to be updated during batch iteration)

    # gpu has too slightly too small memory for loading whole test set -> use cpu
    device = 'cpu'
    model = model.to(device)
    criterion.weight = criterion.weight.to(device)

    y_true_stack_total = torch.Tensor().to(device)
    y_true_stack_selection = torch.Tensor().to(device)
    y_pred_binary_stack_total = torch.Tensor().to(device)
    y_pred_logits_nosnow_stack_selection = torch.Tensor().to(device)

    for x, y in dloader:
        batch_iteration[phase] += 1
        
        # move to CPU (due to memory issues on GPU)
        x = x.to(device)
        y = y.to(device)

        # move to GPU
        # working with whole images
        # x = x.to(device)
        # y = y.to(device)
        y = y.squeeze(1)

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
        # y_pred_logits_data: at idx 0: snow / at idx 1: nosnow
        y_pred_logits_data = y_pred_logits[:, 1:, :, :]  # shape [BATCHSIZE, 2, 256, 256], consider all logits excepts for 0th class (no data) -> consider logits for class 1 and 2 (3)
        y_pred_logits_snow = y_pred_logits_data[:, 0, :, :]  # shape [BS, 256, 256]  # considers logits for snow class
        y_pred_logits_nosnow = y_pred_logits_data[:, 1, :, :]  # shape [BS, 256, 256]  # considers logits for nosnow class

        # ypld = y_pred_logits_data.to('cpu')
        y_pred_binary_data = y_pred_logits_data.argmax(axis=1, keepdim=False)  # only contains 0 (snow) or 1 (no_snow), note: this unfortunately also works if nan values in y_pred_logits_data 

        # save images from test set
        print('saving test images...')


        # TODO: create function or make somehow nicer :)
        x_denormalized = denormalize_fct(x)
        matplotlib.image.imsave(f'segmentation_visualizations/testing/full_y_pred_batchit_{batch_iteration[phase]}_cmap.png', y_pred_binary_data[0].cpu().detach(), cmap=cmap_pred)
        matplotlib.image.imsave(f'segmentation_visualizations/testing/full_y_true_batchit_{batch_iteration[phase]}_cmap.png', y[0].cpu().detach(), cmap=cmap_true)
        plt.imsave(f'segmentation_visualizations/testing/full_x_denorm_batchit_{batch_iteration[phase]}.png', np.transpose(x_denormalized[0].cpu().detach().numpy(),(1,2,0)))


        """
        plt.imsave(f'segmentation_pred_on_testset/large_patches/full_y_pred_batchit_{batch_iteration[phase]}.png', y_pred_binary_data[0].cpu().detach())
        plt.imsave(f'segmentation_pred_on_testset/large_patches/full_y_true_batchit_{batch_iteration[phase]}.png', y[0].cpu().detach())
        plt.imsave(f'segmentation_pred_on_testset/large_patches/full_x_batchit_{batch_iteration[phase]}.png', np.transpose(x[0].cpu().detach().numpy(),(1,2,0)))

        plt.imsave(f'segmentation_pred_on_testset/large_patches/full_y_pred_batchit_{batch_iteration[phase]}.tiff', y_pred_binary_data[0].cpu().detach())
        plt.imsave(f'segmentation_pred_on_testset/large_patches/full_y_true_batchit_{batch_iteration[phase]}.tiff', y[0].cpu().detach())
        plt.imsave(f'segmentation_pred_on_testset/large_patches/full_x_batchit_{batch_iteration[phase]}.tiff', np.transpose(x[0].cpu().detach().numpy(),(1,2,0)))
        """

        # STATS
        y_true_flat = y.flatten()  # contains 0 (nodata), 1 (snow), 2 (nosnow)
        y_pred_logits_snow_flat = y_pred_logits_snow.flatten()
        y_pred_logits_nosnow_flat = y_pred_logits_nosnow.flatten()

        # for metrics, only consider predictions of pixels that are not no_data
        y_true_data = y_true_flat[y_true_flat != 0]  # contains 1, 2
        y_true_data[y_true_data==1] = 0  # convert ones to zeros
        y_true_data[y_true_data==2] = 1  # convert twos to ones
        # y_true_data now contains 0 (snow), 1 (no_snow)

        y_pred_binary_flat = y_pred_binary_data.flatten()  # contains 0 (snow), 1 (nosnow)
        y_pred_binary_data = y_pred_binary_flat[y_true_flat != 0]
        # y_pred_data now contains 0 (snow), 1 (nosnow)

        # remove values at indices where the GT was no data
        y_pred_logits_snow_flat = y_pred_logits_snow_flat[y_true_flat != 0]
        y_pred_logits_nosnow_flat = y_pred_logits_nosnow_flat[y_true_flat != 0]

        y_pred_logits_snow_flat_selection = y_pred_logits_snow_flat[::1000]  # use only every 1000th element, to avoid memory issues
        y_pred_logits_nosnow_flat_selection = y_pred_logits_nosnow_flat[::1000]
        
        # slice GT in same way as the predicted logits, s.t. they match afterwards
        y_true_data_selection = y_true_data[::1000]

        # for all metrics, only log every 1000th element:
        y_true_data = y_true_data[::1000]
        y_pred_binary_data = y_pred_binary_data[::1000]


        y_true_stack_total = torch.cat((y_true_stack_total, y_true_data), 0)  # append to flattened torch tensor 
        y_pred_binary_stack_total = torch.cat((y_pred_binary_stack_total, y_pred_binary_data), 0)  # append to flattened torch tensor 
        y_true_stack_selection = torch.cat((y_true_stack_selection, y_true_data_selection), 0)  # append to flattened torch tensor 
        y_pred_logits_nosnow_stack_selection = torch.cat((y_pred_logits_nosnow_stack_selection, y_pred_logits_nosnow_flat_selection), 0)  # append to flattened torch tensor 

        # losses
        batch_loss = loss.item() * x.shape[0]  # loss of whole batch (loss*batchsize (as loss was averaged ('mean')), each item of batch had this loss on avg)
        running_loss += batch_loss


        if batch_iteration[phase]%len(dloader) == 0:  # after having seen the whole test set, do logging
            loss = running_loss/len(dloader)
            print(f'batch iteration: {batch_iteration[phase]} / {len(dloader)*(epoch+1)} ... {phase} loss (avg over whole test dataloader): {loss}')
            get_and_log_metrics(yt=y_true_stack_total.cpu().detach(), yt_short= y_true_stack_selection.cpu().detach(), ypred_bin=y_pred_binary_stack_total.cpu().detach(), ypred_logits=y_pred_logits_nosnow_stack_selection.cpu().detach(), ep=epoch, batch_it_loss=loss, ph=phase, bi=batch_iteration[phase])

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

            # y_true_total = torch.Tensor().to(device)  # initialize as empty tensor
            # y_pred_binary_total = torch.Tensor().to(device)  # initialize as empty tensor
            # y_pred_logits_nosnow_total = torch.Tensor().to(device)  # only append the data logits

            y_true_stack_total = torch.Tensor().to(device)
            y_true_stack_selection = torch.Tensor().to(device)
            y_pred_binary_stack_total = torch.Tensor().to(device)
            y_pred_logits_nosnow_stack_selection = torch.Tensor().to(device)

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
                y = y.squeeze(1)

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
                # y_pred_logits_data: at idx 0: snow / at idx 1: nosnow
                y_pred_logits_data = y_pred_logits[:, 1:, :, :]  # shape [BATCHSIZE, 2, 256, 256], consider all logits excepts for 0th class (no data) -> consider logits for class 1 and 2 (3)
                y_pred_logits_snow = y_pred_logits_data[:, 0, :, :]  # shape [BS, 256, 256]  # considers logits for snow class
                y_pred_logits_nosnow = y_pred_logits_data[:, 1, :, :]  # shape [BS, 256, 256]  # considers logits for nosnow class

                y_pred_binary_data = y_pred_logits_data.argmax(axis=1, keepdim=False)  # only contains 0 (snow) or 1 (no_snow), note: this unfortunately also works if nan values in y_pred_logits_data 
                # this y_pred_binary_data can be compared to y (GT) (take one element of batch for comparison)

                # save images from validation (if in 2nd but last loop from last epoch)
                # 2nd but last loop to be sure that full batch can be taken (not some residual images)
                if epoch == num_epochs-1 and batch_iteration[phase]%len(dloader) == 1 and batch_iteration[phase] !=1 and phase == 'val':
                    # save patches in last validation loop:
                    print('saving validation images...')
                    # note: after .float(), values would be 0.0, 1.0 and 2.0 -> save_image clip expect input between 0..1 -> clips 2 to 1  # therefore, do y/2 -> get 0.0, 0.5 and 1.0 -> nice plots  # TODO colormap

                    # TODO: create function or make somehow nicer :)
                    x_denormalized = denormalize_fct(x[0])
                    matplotlib.image.imsave('segmentation_visualizations/validation/patch_y_pred_cmap_1.png', y_pred_binary_data[0].cpu().detach(), cmap=cmap_pred)
                    matplotlib.image.imsave('segmentation_visualizations/validation/patch_y_true_cmap_1.png', y[0].cpu().detach(), cmap=cmap_true)
                    plt.imsave(f'segmentation_visualizations/validation/patch_x_denorm_batchit_1.png', np.transpose(x_denormalized.cpu().detach().numpy(),(1,2,0)))

                    x_denormalized = denormalize_fct(x[1])
                    matplotlib.image.imsave('segmentation_visualizations/validation/patch_y_pred_cmap_2.png', y_pred_binary_data[1].cpu().detach(), cmap=cmap_pred)
                    matplotlib.image.imsave('segmentation_visualizations/validation/patch_y_true_cmap_2.png', y[1].cpu().detach(), cmap=cmap_true)
                    plt.imsave(f'segmentation_visualizations/validation/patch_x_denorm_batchit_2.png', np.transpose(x_denormalized.cpu().detach().numpy(),(1,2,0)))

                    x_denormalized = denormalize_fct(x[2])
                    matplotlib.image.imsave('segmentation_visualizations/validation/patch_y_pred_cmap_3.png', y_pred_binary_data[2].cpu().detach(), cmap=cmap_pred)
                    matplotlib.image.imsave('segmentation_visualizations/validation/patch_y_true_cmap_3.png', y[2].cpu().detach(), cmap=cmap_true)
                    plt.imsave(f'segmentation_visualizations/validation/patch_x_denorm_batchit_3.png', np.transpose(x_denormalized.cpu().detach().numpy(),(1,2,0)))

                    x_denormalized = denormalize_fct(x[3])
                    matplotlib.image.imsave('segmentation_visualizations/validation/patch_y_pred_cmap_4.png', y_pred_binary_data[3].cpu().detach(), cmap=cmap_pred)
                    matplotlib.image.imsave('segmentation_visualizations/validation/patch_y_true_cmap_4.png', y[3].cpu().detach(), cmap=cmap_true)
                    plt.imsave(f'segmentation_visualizations/validation/patch_x_denorm_batchit_4.png', np.transpose(x_denormalized.cpu().detach().numpy(),(1,2,0)))

                    """
                    to plot all from same batch in same image: sth like this, but with colormap - ev mpl alternative?
                    for btch in x.shape[0]:
                        f, axarr = plt.subplots(2,2)
                        axarr[0,0].imshow(y_pred_binary_data[0].cpu().detach())
                        axarr[0,1].imshow(y_pred_binary_data[1].cpu().detach())
                        axarr[1,0].imshow(y_pred_binary_data[2].cpu().detach())
                        axarr[1,1].imshow(y_pred_binary_data[3].cpu().detach())
                        plt.savefig('patch_ypred_hstack.png')
                    """



                    """
                    torchvision.utils.save_image(y_pred_binary_data.unsqueeze(1).float(), 'segmentation_visualizations/validation/patch_y_pred.png')
                    torchvision.utils.save_image((y/2.).unsqueeze(1), 'segmentation_visualizations/validation/patch_y_true.png')
                    torchvision.utils.save_image(x.float(), 'segmentation_visualizations/validation/patch_x.png')

                    torchvision.utils.save_image(y_pred_binary_data[0], 'segmentation_visualizations/validation/patch_y_pred_01.png')
                    torchvision.utils.save_image(y[0], 'segmentation_visualizations/validation/patch_y_true_01.png')
                    torchvision.utils.save_image(x[0], 'segmentation_visualizations/validation/patch_x_01.png')

                    torchvision.utils.save_image(y_pred_binary_data[1], 'segmentation_visualizations/validation/patch_y_pred_02.png')
                    torchvision.utils.save_image(y[1], 'segmentation_visualizations/validation/patch_y_true_02.png')
                    torchvision.utils.save_image(x[1], 'segmentation_visualizations/validation/patch_x_02.png')

                    torchvision.utils.save_image(y_pred_binary_data[2], 'segmentation_visualizations/validation/patch_y_pred_03.png')
                    torchvision.utils.save_image(y[2], 'segmentation_visualizations/validation/patch_y_true_03.png')
                    torchvision.utils.save_image(x[2], 'segmentation_visualizations/validation/patch_x_03.png')

                    torchvision.utils.save_image(y_pred_binary_data[3], 'segmentation_visualizations/validation/patch_y_pred_04.png')
                    torchvision.utils.save_image(y[3], 'segmentation_visualizations/validation/patch_y_true_04.png')
                    torchvision.utils.save_image(x[3], 'segmentation_visualizations/validation/patch_x_04.png')
                    """





                    """
                    plt.imsave('segmentation_pred_on_testset/large_patches/patch_y_pred_01.tiff', y_pred_binary_data[0].cpu().detach())
                    plt.imsave('segmentation_pred_on_testset/large_patches/patch_y_true_01.tiff', y[0].cpu().detach())
                    plt.imsave('segmentation_pred_on_testset/large_patches/patch_x_01.tiff', np.transpose(x[0].cpu().detach().numpy(),(1,2,0)))

                    plt.imsave('segmentation_pred_on_testset/large_patches/patch_y_pred_02.tiff', y_pred_binary_data[1].cpu().detach())
                    plt.imsave('segmentation_pred_on_testset/large_patches/patch_y_true_02.tiff', y[1].cpu().detach())
                    plt.imsave('segmentation_pred_on_testset/large_patches/patch_x_02.tiff', np.transpose(x[1].cpu().detach().numpy(),(1,2,0)))

                    plt.imsave('segmentation_pred_on_testset/large_patches/patch_y_pred_03.tiff', y_pred_binary_data[2].cpu().detach())
                    plt.imsave('segmentation_pred_on_testset/large_patches/patch_y_true_03.tiff', y[2].cpu().detach())
                    plt.imsave('segmentation_pred_on_testset/large_patches/patch_x_03.tiff', np.transpose(x[2].cpu().detach().numpy(),(1,2,0)))

                    plt.imsave('segmentation_pred_on_testset/large_patches/patch_y_pred_04.tiff', y_pred_binary_data[3].cpu().detach())
                    plt.imsave('segmentation_pred_on_testset/large_patches/patch_y_true_04.tiff', y[3].cpu().detach())
                    plt.imsave('segmentation_pred_on_testset/large_patches/patch_x_04.tiff', np.transpose(x[3].cpu().detach().numpy(),(1,2,0)))
                    """


                # STATS
                y_true_flat = y.flatten()  # contains 0 (nodata), 1 (snow), 2 (nosnow)
                y_pred_logits_snow_flat = y_pred_logits_snow.flatten()
                y_pred_logits_nosnow_flat = y_pred_logits_nosnow.flatten()

                # for metrics, only consider predictions of pixels that are not no_data
                y_true_data = y_true_flat[y_true_flat != 0]  # contains 1, 2
                y_true_data[y_true_data==1] = 0  # convert ones to zeros
                y_true_data[y_true_data==2] = 1  # convert twos to ones
                # y_true_data now contains 0 (snow), 1 (no_snow)

                y_pred_binary_flat = y_pred_binary_data.flatten()  # contains 0 (snow), 1 (nosnow)
                y_pred_binary_data = y_pred_binary_flat[y_true_flat != 0]
                # y_pred_data now contains 0 (snow), 1 (nosnow)

                # remove values at indices where the GT was no data
                y_pred_logits_snow_flat = y_pred_logits_snow_flat[y_true_flat != 0]
                y_pred_logits_nosnow_flat = y_pred_logits_nosnow_flat[y_true_flat != 0]

                y_pred_logits_snow_flat_selection = y_pred_logits_snow_flat[::1000]  # use only every 1000th element, to avoid memory issues
                y_pred_logits_nosnow_flat_selection = y_pred_logits_nosnow_flat[::1000]
                
                # slice GT in same way as the predicted logits, s.t. they match afterwards
                y_true_data_selection = y_true_data[::1000]

                # for all metrics, only log every 1000th element:
                y_true_data = y_true_data[::1000]
                y_pred_binary_data = y_pred_binary_data[::1000]


                y_true_stack_total = torch.cat((y_true_stack_total, y_true_data), 0)  # append to flattened torch tensor 
                y_pred_binary_stack_total = torch.cat((y_pred_binary_stack_total, y_pred_binary_data), 0)  # append to flattened torch tensor 
                y_true_stack_selection = torch.cat((y_true_stack_selection, y_true_data_selection), 0)  # append to flattened torch tensor 
                y_pred_logits_nosnow_stack_selection = torch.cat((y_pred_logits_nosnow_stack_selection, y_pred_logits_nosnow_flat_selection), 0)  # append to flattened torch tensor 


                # LOSSES
                batch_loss = loss.item() * x.shape[0]  # loss of whole batch (loss*batchsize (as loss was averaged ('mean')), each item of batch had this loss on avg)
                running_loss += batch_loss

                if phase == 'train':
                    if (batch_iteration[phase]%LOG_EVERY) == 0:
                        loss = running_loss/LOG_EVERY
                        print(f'batch iteration: {batch_iteration[phase]} / {len(dloader)*(epoch+1)} with {phase} loss (avg over these {LOG_EVERY} batch iterations): {loss}')
                        get_and_log_metrics(yt=y_true_stack_total.cpu().detach(), yt_short= y_true_stack_selection.cpu().detach(), ypred_bin=y_pred_binary_stack_total.cpu().detach(), ypred_logits=y_pred_logits_nosnow_stack_selection.cpu().detach(), ep=epoch, batch_it_loss=loss, ph=phase, bi=batch_iteration[phase])
                        running_loss = 0

                if phase == 'val':
                    if batch_iteration[phase]%len(dloader) == 0:
                        loss = running_loss / len(dloader)
                        print(f'batch iteration: {batch_iteration[phase]} / {len(dloader)*(epoch+1)} ... {phase} loss (avg over whole validation dataloader): {loss}')
                        get_and_log_metrics(yt=y_true_stack_total.cpu().detach(), yt_short= y_true_stack_selection.cpu().detach(), ypred_bin=y_pred_binary_stack_total.cpu().detach(), ypred_logits=y_pred_logits_nosnow_stack_selection.cpu().detach(), ep=epoch, batch_it_loss=loss, ph=phase, bi=batch_iteration[phase])
                        # as we're in last loop for validation, running_loss will be set to 0 anyways (changing the phase back to train)

            if phase == 'train':  # at end of epoch (training, could also be end of validation)
                if scheduler is not None:
                    scheduler.step()
                    # scheduler.print_lr()

        print()  # end of epoch
    
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
BETA = 1
EPSILON = 0
PATH_MODEL = f'models/segmentation/{STATIONS_CAM_LST}_bs_{BATCH_SIZE}_LR_{LEARNING_RATE}_epochs_{EPOCHS}_lr_sched_{LR_SCHEDULER}_optim_{OPTIM}'

#PATH_LOAD_MODEL = f'final_models_classification_v01/{STATIONS_CAM_LST}_bs_{BATCH_SIZE}_LR_{LEARNING_RATE}_epochs_{EPOCHS}_weighted_{WEIGHTED}_lr_sched_{LR_SCHEDULER}'
PATH_LOAD_MODEL = PATH_MODEL
LOAD_MODEL = False

LOG_EVERY = 20

clist_pred = [(0, 'white'), (1, 'green')]
clist_true = [(0, 'red'), (1./2., 'white'), (2./2., 'green')]

cmap_pred = matplotlib.colors.LinearSegmentedColormap.from_list('name', clist_pred)
cmap_true = matplotlib.colors.LinearSegmentedColormap.from_list('name', clist_true)

mean_ImageNet = np.asarray([0.485, 0.456, 0.406])
std_ImageNet = np.asarray([0.229, 0.224, 0.225])
denormalize_fct = torchvision.transforms.Normalize((-1*mean_ImageNet/std_ImageNet), (1.0/std_ImageNet))

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
dloader_test = DataLoader(dataset=dset_test, batch_size=1)  # TODO: ev change to 1 (if memory issues)

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

# model = SimpleNetwork(in_channels=3, classes=N_CLASSES)


if LOAD_MODEL:
    if os.path.exists(PATH_LOAD_MODEL) == True:
        print('trained model already exists, loading model...')
        model = torch.load(PATH_LOAD_MODEL)


model = model.to(device)
# model = model.to(torch.float64)  # set higher precision

"""
# does not really work TODO ev delete
preprocess_input = get_preprocessing_fn('resnet34', pretrained='imagenet')
x_try = torch.randn(size=(8, 3, 512, 512), dtype=torch.float32)
x_try = preprocess_input(x_try)
out = model(x_try)
"""

# loss functions to try: BCE / IoU-loss / focal loss
# criterion = nn.CrossEntropyLoss(weight=torch.Tensor([0, 0.4, 0.6]).to(device), reduction='mean')
# use normal precision 
criterion = nn.CrossEntropyLoss(weight=torch.Tensor([0, 0.2, 0.8]).to(device), reduction='mean')
criterion = nn.CrossEntropyLoss(weight=torch.Tensor([0, 1, 1]).to(device), reduction='mean')

if OPTIM == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
elif OPTIM == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)  # weight_decay: It is used for adding the l2 penality to the loss (default = 0)

if LR_SCHEDULER != 'None':
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=int(LR_SCHEDULER), gamma=0.4)  # Decay LR by a factor of gamma every 'step_size' epochs
elif LR_SCHEDULER == 'None':
    exp_lr_scheduler = None

print('start training / validation...')

if not LOAD_MODEL:
    train_val_model(model=model, criterion=criterion, optimizer=optimizer, scheduler=exp_lr_scheduler, num_epochs=EPOCHS)
    print('optimal threshold (to be used for test set): ', OPTIMAL_THRESHOLD)

print('start testing model...')


test_model(model=model)

print()