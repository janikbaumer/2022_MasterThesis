import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader

# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
'''
class Binary_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        """
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        """

    def forward(self, x):
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        """
        return x
'''

"""
# https://towardsdatascience.com/pytorch-vision-binary-image-classification-d9a227705cf9
class HotDogClassifier(nn.Module):
    def __init__(self):
        super(HotDogClassifier, self).__init__()
        self.block1 = self.conv_block(c_in=3, c_out=256, dropout=0.1, kernel_size=5, stride=1, padding=2)
        self.block2 = self.conv_block(c_in=256, c_out=128, dropout=0.1, kernel_size=3, stride=1, padding=1)
        self.block3 = self.conv_block(c_in=128, c_out=64, dropout=0.1, kernel_size=3, stride=1, padding=1)
        self.lastcnn = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=56, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        x = self.block1(x)
        x = self.maxpool(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.maxpool(x)
        x = self.lastcnn(x)
        return x
    
    def conv_block(self, c_in, c_out, dropout,  **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )
        return seq_block
"""

# define the CNN architecture
class MyNet(nn.Module):
   def __init__(self):
       super(MyNet, self).__init__()
       self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
       self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
       self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
       # ev add: self.conv2_drop = nn.Dropout2d()

       self.pool = nn.MaxPool2d(2, 2)
       self.fc1 = nn.Linear(64*4*4*4*4, 64)
       self.fc2 = nn.Linear(64, 1)
       self.dropout = nn.Dropout(0.1)

   def forward(self, x):
       # add sequence of convolutional and max pooling layers
       x = self.pool(F.relu(self.conv1(x)))
       x = self.pool(F.relu(self.conv2(x)))
       x = self.pool(F.relu(self.conv3(x)))
       x = x.view(-1, 64 * 4 * 4*4*4)
       x = self.dropout(x)
       x = F.relu(self.fc1(x))
       x = self.dropout(x)
       x = F.relu(self.fc2(x))
       x = F.sigmoid(x)
       return x
