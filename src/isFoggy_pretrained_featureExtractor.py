from time import time
from numpy import datetime_as_string
import torch
from dischma_set import DischmaSet
from torch.utils.data import DataLoader
from binary_classifier import MyNet # , Binary_Classifier
from torch import nn
from torchvision import models
import torchvision
from torch.utils.data import random_split
import matplotlib.pyplot as plt
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


torch.seed()
torch.manual_seed(42)

BATCH_SIZE = 8
LEARNING_RATE = 0.01
EPOCHS = 15
N_CLASSES = 2
TRAIN_SPLIT = 0.8

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dset_bb1 = DischmaSet(root='../datasets/dataset_downsampled/', station='Buelenberg', camera=1)
print('Dischma set (BB1) created.')

dset_bb1_train, dset_bb1_val = get_train_val_split(dset_bb1)

dloader_train = DataLoader(dataset=dset_bb1_train, batch_size=BATCH_SIZE)
dloader_val = DataLoader(dataset=dset_bb1_val, batch_size=BATCH_SIZE)


def train_model(model, criterion, optimizer, scheduler, num_epochs):
    print('start training model...')
    since = time()

    for epoch in range(num_epochs):
        print(f'{epoch}/{num_epochs}')
        print('-' * 10)
        
        """
        for phase in ['train','val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
        """
        
        running_loss = 0.0  # loss (to be updated during loop)
        running_corrects = 0  # correct labels (to be updated during loop)
        
        train_loss, epoch_loss = 0, 0

        loop = 0
        print('start training...')
        for x, y in dloader_train:  # x,y are already moved to device in dataloader
            loop += 1

            """
            # plot at nth loop
            if loop == 26:
                print_grid(x,y)
            """    

            x = x.to(device)
            y = y.to(device)
            y = y.float()

            optimizer.zero_grad()  # gradients do not have to be kept from last step

            # TODO: ev add: track history only if in train
            # with torch.set_grad_enabled(phase == 'train'):

            pred_probab = model(x)  # fwd pass
            pred_probab_class_1 = pred_probab[:, 1]
            #prediction = pred_probab.argmax(dim=1)
            #prediction = prediction.float()

            # loss = criterion(prediction, y)  # loss is calculated batchsize times, and then averaged
            loss = criterion(pred_probab_class_1, y)
            # TODO calc metrics (f1 etc)

            loss.backward()  # backprop
            optimizer.step()  # update params
            train_loss += loss.item() * x.size(0)  # loss*batchsize (as loss was averaged ('mean'))
            epoch_loss += loss.item() * x.size(0)

            if (loop%200) == 0:
                print('loop: ', loop, '/', len(dloader_train), ' ... ', 'train loss (avg over these 100 loops): ', train_loss/200)
                train_loss = 0
        
        print('epoch loss = ', epoch_loss/len(dloader_train))

model = models.resnet18(pretrained=True)
#model = model.to(device)

# do not apply backprop on weight that were used for feature extraction
for param in model.parameters():
    param.requires_grad = False
    param = param.to(device)

# on these, backprop will be done
n_features = model.fc.in_features
model.fc = nn.Linear(n_features, N_CLASSES)  # .to(device)  # modify fully connected layer (last layer)
model = nn.Sequential(model, nn.Softmax(dim=None))
model = model.to(device)


#criterion = nn.CrossEntropyLoss(reduction='mean')
criterion = nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr= LEARNING_RATE)  # TODO ev add momentum

train_model(model=model, criterion=criterion, optimizer=optimizer, scheduler=None, num_epochs=EPOCHS)

# ev add (TODO)
# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

