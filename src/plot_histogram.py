import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

path_image = '/scratch2/jbaumer/2022_MasterThesis/datasets/dataset_complete/DischmaCams/Buelenberg/1/20200101090501.jpg'

img = cv.imread(path_image)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig(fname='histogram.png')