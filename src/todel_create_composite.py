"""
input: a certain day
    for this day, load all images
    check the images with the trained model - only keep the ones with label 0
    for these images, create averaged label
    save composite image
    !!! start with development dataset (downsampled)
"""

import os
import torch


def load_image():
    pass


BASE_PATH = 'datasets/'
MODEL = 'Stillberg2_bs_8_LR_0.01_epochs_20_weighted_True'

model = torch.load(os.path.join('models', MODEL))



print(type(model))
