import torch
import torchvision
import argparse
import matplotlib.pyplot as plt

from time import time
from numpy import datetime_as_string
from dischma_set import DischmaSet
from torch.utils.data import DataLoader
from binary_classifier import MyNet # , Binary_Classifier
from torch import nn
from torchvision import models
from torch.utils.data import random_split
from torch.optim import lr_scheduler
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score

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
    dset_train, dset_val = random_split(dset_bb1, [len_train, len_val])  # Split Pytorch tensor
    return dset_train, dset_val

def print_grid(x,y, batchsize):
    x = x.cpu()
    y = y.cpu()
    print(y)
    grid_img = torchvision.utils.make_grid(x, nrow=batchsize/2, normalize=True)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()

def train_model(model, dloader, criterion, optimizer, scheduler, num_epochs):
    print('start TRAINING model...')
    model = model.train()
    train_since = time()

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
            pred_probab_class_1 = pred_probab[:, 1]
            prediction_binary = pred_probab.argmax(dim=1).float()

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



            if (loop%200) == 0:
                print('loop: ', loop, '/', len(dloader_train), ' ... ', 'train loss (avg over these 200 loops): ', train_loss/200)
                train_loss = 0

        print('epoch loss = ', epoch_loss/len(dloader_train))

        cm, accuracy, precision, recall, f1 = get_and_print_stats(yt=all_y_true, yp=all_y_pred, mode='train')
    # saving model
    torch.save(obj=model, f=PATH_MODEL)
    print('training time in seconds: ', time() - train_since)


def val_model(model, dloader, criterion):
    print('start VALIDATING the model...')
    model = model.eval()
    val_since = time()

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

    print('validation time in seconds: ', time() - val_since)


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

args = parser.parse_args()

STATION = args.station
CAM = int(args.cam)
BATCH_SIZE = int(args.batch_size)
LEARNING_RATE = float(args.lr)
EPOCHS = int(args.epochs)
TRAIN_SPLIT = float(args.train_split)

N_CLASSES = 2
PATH_MODEL = f'models/{STATION}{CAM}_bs_{BATCH_SIZE}_LR_{LEARNING_RATE}_epochs_{EPOCHS}'


# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create datasets and dataloaders
dset_bb1 = DischmaSet(root='../datasets/dataset_downsampled_devel/', station='Buelenberg', camera=1)
print(f'Dischma set {STATION}{CAM} created.')
dset_bb1_train, dset_bb1_val = get_train_val_split(dset_bb1)
dloader_train = DataLoader(dataset=dset_bb1_train, batch_size=BATCH_SIZE)
dloader_val = DataLoader(dataset=dset_bb1_val, batch_size=BATCH_SIZE)

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
criterion = nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr= LEARNING_RATE)  # TODO ev add momentum


"""
# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)  # Decay LR by a factor of 0.1 every 7 epochs
"""

train_model(model=model, dloader=dloader_train, criterion=criterion, optimizer=optimizer, scheduler=None, num_epochs=EPOCHS)
val_model(model=model, dloader=dloader_val, criterion=criterion)

