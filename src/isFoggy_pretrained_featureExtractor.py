from time import time
import torch
from dischma_set import DischmaSet
from torch.utils.data import DataLoader
from binary_classifier import MyNet # , Binary_Classifier
from torch import nn
from torchvision import models


BATCH_SIZE = 8
LEARNING_RATE = 0.01
EPOCHS = 3
N_CLASSES = 1

print('imports done')
dset_train = DischmaSet(root='../datasets/dataset_downsampled/')
print('dischma set created')
dloader_train = DataLoader(dataset=dset_train, batch_size=BATCH_SIZE)

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time()

    for epoch in range(num_epochs):
        print(f'{epoch}/{num_epochs}')
        print('-' * 10)

        for phase in ['train','val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0  # loss (to be updated during loop)
            running_corrects = 0  # correct labels (to be updated during loop)

        for x, y in dloader_train:  # x,y are already moved to device in dataloader
            x = x.to(device)
            y = y.to(device)

            print(x.shape)
            print(y.shape)
            print(x.dtype)
            print(y.dtype)

            optimizer.zero_grad()  # gradients do not have to be kept from last step
            
            # TODO: ev add: track history if only in train
            # with torch.set_grad_enabled(phase == 'train'):
            
            prediction = model(x)  # fwd pass

            loss = criterion(prediction, y)
            loss.backward()  # backprop
            optimizer.step()  # update params
            train_loss += loss.item() * x.size(0)
            


model = models.resnet18(pretrained=True)
model.to(device)

# do not apply backprop on weight that were used for feature extraction
for param in model.parameters():
    param.requires_grad = False

# on these, backprop will be done
n_features = model.fc.in_features
model.fc = nn.Linear(n_features, N_CLASSES)  # modify fully connected layer (last layer)
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr= LEARNING_RATE)  # TODO ev add momentum
#optimizer = torch.optim.Adam(model.parameters(), lr= LEARNING_RATE)

print('start training model...')
train_model(model=model, criterion=criterion, optimizer=optimizer, scheduler=None, num_epochs=1)

# ev add (TODO)
# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
