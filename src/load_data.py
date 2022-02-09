import numpy
import torch
import os

print('sucess')

PATH_ORIG_IMGS = '../datasets/ExampleData/Orig_images'
print(os.getcwd())


# Python program to read
# image using PIL module

# importing PIL
from PIL import Image
# Read image
img = Image.open(f'{PATH_ORIG_IMGS}/BB1/20191025083501.jpg')

# prints format of image
print(img.format)

# prints mode of image
print(img.mode)  # JPEG

# Output Images
img.show()  # RGB
