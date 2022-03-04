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
from sklearn.metrics import f1_score, confusion_matrix

print('imports done')


################# FUNCTIONS ######################

def get_train_val_split(dset_full):
    print('splitting in train/test...')
    len_full = len(dset_full)
    len_train = int(TRAIN_SPLIT*len_full)
    len_val = len_full - len_train
    dset_train, dset_val = random_split(dset_bb1, [len_train, len_val])  # Split Pytorch tensor
    return dset_train, dset_val

def print_grid(x,y):
    print(y)
    grid_img = torchvision.utils.make_grid(x, nrow=4, normalize=True)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    print('start training model...')
    model = model.train()
    since = time()

    for epoch in range(num_epochs):
        print(f'{epoch}/{num_epochs}')
        print('-' * 10)

        running_loss = 0.0  # loss (to be updated during loop)
        running_corrects = 0  # correct labels (to be updated during loop)
        train_loss, epoch_loss = 0, 0
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
                print_grid(x,y)
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
            epoch_loss += loss.item() * x.size(0)

            # add sth like: running_corrects += torch.sum(preds == labels.data)


            if (loop%200) == 0:
                print('loop: ', loop, '/', len(dloader_train), ' ... ', 'train loss (avg over these 100 loops): ', train_loss/200)
                train_loss = 0

        print('epoch loss = ', epoch_loss/len(dloader_train))
    
    print('training time in seconds: ', time()-since)


################## REPRODUCIBILITY ##################

torch.seed()
torch.manual_seed(42)


############ ARGPARSERS, GLOBAL VARIABLES ############

parser = argparse.ArgumentParser(description='Run pretrained finetuning.')
parser.add_argument('--batch_size', help='batch size')
parser.add_argument('--lr', help='learning rate')
parser.add_argument('--epochs', help='number of training epochs')
parser.add_argument('--train_split', help='train split')

args = parser.parse_args()

BATCH_SIZE = int(args.batch_size)
LEARNING_RATE = float(args.lr)
EPOCHS = int(args.epochs)
TRAIN_SPLIT = float(args.train_split)

N_CLASSES = 2


# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# create datasets and dataloaders
dset_bb1 = DischmaSet(root='../datasets/dataset_downsampled/', station='Buelenberg', camera=1)
print('Dischma set (BB1) created.')
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



train_model(model=model, criterion=criterion, optimizer=optimizer, scheduler=None, num_epochs=EPOCHS)


