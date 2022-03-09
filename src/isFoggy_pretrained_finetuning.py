import torch
import torchvision
import argparse
import matplotlib.pyplot as plt

from time import time
from numpy import datetime_as_string, rec
from dischma_set import DischmaSet
from torch.utils.data import DataLoader
from binary_classifier import MyNet # , Binary_Classifier
from torch import nn
from torchvision import models
from torch.utils.data import random_split
from torch.optim import lr_scheduler
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
import json
import os
import wandb


print('imports done')


################# FUNCTIONS ######################

def get_and_print_stats(yt, yp, mode=None):
    if mode == 'train':
        print('TRAINING METRICS: ')
    elif mode == 'val':
        print('VALIDATION METRICS: ')

    cm = confusion_matrix(y_true=yt, y_pred=yp)
    print('confusion matrix: \n', cm, '\n')

    accuracy = accuracy_score(y_true=yt, y_pred=yp)
    print('accuracy \n', accuracy, '\n')

    precision = precision_score(y_true=yt, y_pred=yp)
    print('precision score \n', precision, '\n')

    recall = recall_score(y_true=yt, y_pred=yp)
    print('recall score \n', recall, '\n')

    f1 = f1_score(y_true=yt, y_pred=yp)
    print('f1 score \n', f1, '\n')

    return cm, accuracy, precision, recall, f1


def get_train_val_split(dset_full):
    print('splitting in train/test...')
    len_full = len(dset_full)
    len_train = int(TRAIN_SPLIT*len_full)
    len_val = len_full - len_train
    dset_train, dset_val = random_split(dset_full, [len_train, len_val])  # Split Pytorch tensor
    return dset_train, dset_val

def print_grid(x,y, batchsize):
    x = x.cpu()
    y = y.cpu()
    print(y)
    grid_img = torchvision.utils.make_grid(x, nrow=int(batchsize/2), normalize=True)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()

def train_model(model, dloader, criterion, optimizer, scheduler, num_epochs):
    print('start TRAINING model...')
    model = model.train()
    train_since = time()
    all_stats = {}

    for epoch in range(num_epochs):
        print(f'{epoch}/{num_epochs}')
        print('-' * 10)

        running_loss, running_corrects = 0.0, 0  # loss (to be updated during loop)
        train_loss, epoch_loss = 0, 0
        all_y_true, all_y_pred = [], []  # after a complete epoch, these two lists will have same length as dset_train
        loop = 0

        print('start training...')
        for x, y in dloader_train:  # x,y are already moved to device in dataloader
            # move to GPU
            x = x.to(device)
            y = y.to(device)
            y = y.float()

            loop += 1

            """
            # plot at nth loop
            if loop == 26:
                print_grid(x,y, BATCH_SIZE)
            """

            optimizer.zero_grad()  # gradients do not have to be kept from last step

            # TODO: ev add: track history only if in train  --  with torch.set_grad_enabled(phase == 'train'):

            pred_probab = model(x)  # fwd pass
            pred_probab_class_0 = pred_probab[:, 0]
            pred_probab_class_1 = pred_probab[:, 1]
            prediction_binary = pred_probab.argmax(dim=1).float()
            # note: if changed to pred_probab_class_0, then also weights tensor given to criterion must be changed !!!
            loss = criterion(pred_probab_class_1, y)  # loss is calculated batchsize times, and then averaged
            loss.backward()  # backprop
            optimizer.step()  # update params

            # statistics
            running_loss += loss.item() * x.size(0)  # loss*batchsize (as loss was averaged ('mean')), each item of batch had this loss on avg
            running_corrects += torch.sum(prediction_binary == y).item()  #  correct predictions (of this batch)

            y_true = y.cpu().tolist()
            y_pred = prediction_binary.cpu().tolist()

            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)

            epoch_loss += loss.item() * x.size(0)
            train_loss += loss.item() * x.size(0)



            if (loop%200) == 0:
                print('loop: ', loop, '/', len(dloader_train), ' ... ', 'train loss (avg over these 200 loops): ', train_loss/200)
                train_loss = 0

        print('epoch loss = ', epoch_loss/len(dloader_train))
        
        cm, accuracy, precision, recall, f1 = get_and_print_stats(yt=all_y_true, yp=all_y_pred, mode='train')

        all_stats[f'epoch_{epoch}'] = {}

        all_stats[f'epoch_{epoch}']['epoch_loss'] = epoch_loss/len(dloader_train)

        # array([[a,b],
        #        [c,d]])
        #
        # --> [[a,b],[c,d]]
        # cm is a np.array, convert to list of lists to store in dict (cm can be recreated by cm = np.array(cm.tolist()) )
        all_stats[f'epoch_{epoch}']['cm'] = cm.tolist()

        all_stats[f'epoch_{epoch}']['accuracy'] = float(accuracy)
        all_stats[f'epoch_{epoch}']['precision'] = float(precision)
        all_stats[f'epoch_{epoch}']['recall'] = float(recall)
        all_stats[f'epoch_{epoch}']['f1'] = float(f1)
        
        wandb.log({"epoch_loss": epoch_loss/len(dloader_train)})
        wandb.log({'train accuracy' : accuracy})
        wandb.log({'train precision' : precision})
        wandb.log({'train recall' : recall})

        print()
    train_end = time()
    training_time = train_end - train_since
    print('training time in seconds: ', training_time)
    all_stats['training_time'] = training_time

    # saving model
    torch.save(obj=model, f=PATH_MODEL)

    # save dict with statistics
    with open(PATH_STATS_TRAIN, 'w') as fp:
        json.dump(all_stats, fp)


def val_model(model, dloader, criterion):
    print('start VALIDATING the model...')
    model = model.eval()
    val_since = time()
    all_stats = {}

    print('-' * 10)

    running_loss, running_corrects = 0.0, 0  # loss (to be updated during loop)
    train_loss, epoch_loss = 0, 0
    all_y_true, all_y_pred = [], []  # after a complete epoch, these two lists will have same length as dset_train

    loop = 0

    print('start validation loop...')
    with torch.no_grad():
        for x, y in dloader:  # x,y are already moved to device in dataloader
            # move to GPU
            x = x.to(device)
            y = y.to(device)
            y = y.float()
            loop += 1

            """
            if loop == 26:
                print_grid(x, y, BATCH_SIZE)
            """

            pred_probab = model(x)  # fwd pass
            pred_probab_class_1 = pred_probab[:, 1]
            prediction_binary = pred_probab.argmax(dim=1).float()

            loss = criterion(pred_probab_class_1, y)  # loss is calculated batchsize times, and then averaged
            #loss.backward()  # backprop
            #optimizer.step()  # update params

            # statistics
            running_loss += loss.item() * x.size(0)  # loss*batchsize (as loss was averaged ('mean')), each item of batch had this loss on avg
            running_corrects += torch.sum(prediction_binary == y).item()  #  correct predictions (of this batch)

            y_true = y.cpu().tolist()
            y_pred = prediction_binary.cpu().tolist()

            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            
            epoch_loss += loss.item() * x.size(0)


        print('epoch loss = ', epoch_loss/len(dloader_train))

    cm, accuracy, precision, recall, f1 = get_and_print_stats(yt=all_y_true, yp=all_y_pred, mode='val')
    all_stats['validation'] = {}
    all_stats[f'validation']['cm'] = cm.tolist()
    all_stats[f'validation']['accuracy'] = float(accuracy)
    all_stats[f'validation']['precision'] = float(precision)
    all_stats[f'validation']['recall'] = float(recall)
    all_stats[f'validation']['f1'] = float(f1)

    wandb.log({"val epoch_loss": epoch_loss/len(dloader_train)})
    wandb.log({'val accuracy' : accuracy})
    wandb.log({'val precision' : precision})
    wandb.log({'val recall' : recall})

    val_end = time()
    validation_time = val_end - val_since
    print('validation time in seconds: ', validation_time)
    all_stats['validation_time'] = validation_time

    # save dict with statistics
    with open(PATH_STATS_VAL, 'w') as fp:
        json.dump(all_stats, fp)


################## REPRODUCIBILITY ##################

torch.seed()  # only for CPU
torch.manual_seed(42)  # works for CPU and CUDA


############ ARGPARSERS, GLOBAL VARIABLES ############

parser = argparse.ArgumentParser(description='Run pretrained finetuning.')
parser.add_argument('--batch_size', help='batch size')
parser.add_argument('--lr', help='learning rate')
parser.add_argument('--epochs', help='number of training epochs')
parser.add_argument('--train_split', help='train split')
parser.add_argument('--station', help='station')
parser.add_argument('--cam', help='camera number')
parser.add_argument('--weighted', help='whether loss function should be weighted (inversely proportional to occurance')

args = parser.parse_args()

STATION = args.station
CAM = int(args.cam)
BATCH_SIZE = int(args.batch_size)
LEARNING_RATE = float(args.lr)
EPOCHS = int(args.epochs)
TRAIN_SPLIT = float(args.train_split)
WEIGHTED = args.weighted

N_CLASSES = 2
PATH_DATASET = f'../datasets/dataset_downsampled/'
PATH_MODEL = f'models/{STATION}{CAM}_bs_{BATCH_SIZE}_LR_{LEARNING_RATE}_epochs_{EPOCHS}_weighted_{WEIGHTED}'
PATH_STATS_TRAIN = f'stats/{STATION}{CAM}_bs_{BATCH_SIZE}_LR_{LEARNING_RATE}_epochs_{EPOCHS}_weighted_{WEIGHTED}.json'
PATH_STATS_VAL = f'stats/{STATION}{CAM}_bs_{BATCH_SIZE}_LR_{LEARNING_RATE}_epochs_{EPOCHS}_weighted_{WEIGHTED}_validation.json'

wandb.config = {
  "learning_rate": LEARNING_RATE,
  "epochs": EPOCHS,
  "batch_size": BATCH_SIZE
}
wandb.init(project="isFoggy_pretrained_finetuning", entity="jbaumer")


print('path exists? ', os.path.exists(PATH_MODEL))
print(PATH_MODEL)
# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create datasets and dataloaders
dset = DischmaSet(root=PATH_DATASET, station=STATION, camera=CAM)
print(f'Dischma set {STATION}{CAM} created.')
dset_train, dset_val = get_train_val_split(dset)
dloader_train = DataLoader(dataset=dset_train, batch_size=BATCH_SIZE)
dloader_val = DataLoader(dataset=dset_val, batch_size=BATCH_SIZE)

model = models.resnet18(pretrained=True)

# train all layers
for param in model.parameters():
    param.requires_grad = True
    param = param.to(device)

# adapt final layer
n_features = model.fc.in_features
model.fc = nn.Linear(n_features, N_CLASSES)
model = nn.Sequential(model, nn.Softmax(dim=None))
model = model.to(device)


#criterion = nn.CrossEntropyLoss(reduction='mean')
n_class_0, n_class_1 = dset.get_balancedness()
n_tot = n_class_0 + n_class_1
w0, w1 = n_class_1/n_tot, n_class_0/n_tot
weights = torch.Tensor([w0, w1]).to(device)

if WEIGHTED == 'False':
    criterion = nn.BCELoss(reduction='mean', weight=None)
elif WEIGHTED == 'True':
    criterion = nn.BCELoss(reduction='mean', weight=weights[1])  # TODO: currently, all occurances are considered, optimal would be to only consider occ. of train split


optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # TODO ev add momentum


"""
# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)  # Decay LR by a factor of 0.1 every 7 epochs
"""

if os.path.exists(PATH_MODEL) == True:
    print('trained model already exists, loading model...')
    del model
    model = torch.load(PATH_MODEL)
else:
    train_model(model=model, dloader=dloader_train, criterion=criterion, optimizer=optimizer, scheduler=None, num_epochs=EPOCHS)

print()
print()
print()
val_model(model=model, dloader=dloader_val, criterion=criterion)
