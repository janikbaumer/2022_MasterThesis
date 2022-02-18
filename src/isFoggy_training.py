import torch
from dischma_set import DischmaSet
from torch.utils.data import DataLoader
from binary_classifier import MyNet # , Binary_Classifier
from torch import nn


BATCH_SIZE = 8
LEARNING_RATE = 0.01
EPOCHS = 3

dset_train = DischmaSet()
dloader_train = DataLoader(dataset=dset_train, batch_size=BATCH_SIZE)

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = MyNet()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr= LEARNING_RATE)

train_losses = []

for epoch in range(EPOCHS):
    train_loss = 0.0
    model.train()

    for x, y in dloader_train:  # x,y are already moved to device in dataloader
        x = x.to(device)
        y = y.to(device)

        print(x.shape)
        print(y.shape)
        print(x.dtype)
        print(y.dtype)
        optimizer.zero_grad()  # gradients do not have to be kept from last step
        prediction = model(x)  # fwd pass
        loss = criterion(prediction, y)
        loss.backward()  # backprop
        optimizer.step()  # update params
        train_loss += loss.item() * x.size(0)
