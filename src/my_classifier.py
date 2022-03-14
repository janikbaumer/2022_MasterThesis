efficientnet_b0 = models.efficientnet_b0()


import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader


# define the CNN architecture
class MyNet(nn.Module):

   def __init__(self):
       super(MyNet, self).__init__()
       self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
       self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
       self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
       self.pool = nn.MaxPool2d(2, 2)

       # ev add: self.conv2_drop = nn.Dropout2d()
       self.dropout = nn.Dropout(0.1)

       self.fc1 = nn.Linear(64*73*48, 64)
       self.fc2 = nn.Linear(64, 1)


   def forward(self, x):
       # add sequence of convolutional and max pooling layers
       x = self.conv1(x)
       x = F.relu(x)
       x = self.pool(x)

       x = self.conv2(x)
       x = F.relu(x)
       x = self.pool(x)

       x = self.conv3(x)
       x = F.relu(x)
       x = self.pool(x)

       # x = self.pool(F.relu(self.conv1(x)))
       # x = self.pool(F.relu(self.conv2(x)))
       # x = self.pool(F.relu(self.conv3(x)))
       
       x = x.view(x.shape[0], -1)
       # x = self.dropout(x)

       x = F.relu(self.fc1(x))
       x = self.dropout(x)
       x = F.relu(self.fc2(x))
       x = F.sigmoid(x)
       return x
