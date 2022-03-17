#!/usr/bin/env python

from PIL import Image
from matplotlib.image import imread
from matplotlib.pyplot import imsave
import glob
import numpy as np
import matplotlib.pyplot as plt
import datetime
import sys

########## Define variable ################
todaystr = sys.argv[1]
folder_in = sys.argv[2]
folder_out = sys.argv[3]

#today = datetime.datetime.now()
#todaystr = datetime.datetime.strftime(today,'%Y%m%d')
#todaystr='20200416'

print('Processing ',folder_out)
print('Processing images from ', todaystr)

########## get threshold values from text file ################
thresholdfile = folder_out+'thresholds.txt' 		#file name with threshold data

f = open(thresholdfile,'r')
A0_min=[]; index_min=[]; a=[]

#loop through file and read data
for line in f: 	
	a=line.split()
	A0_min.append(float(a[0]))
	index_min.append(float(a[1]))
	
f.close()
A0_min=np.array(A0_min)
index_min=np.array(index_min)

########## get image data ################
foggyfile = folder_out+todaystr+'.txt' 		#file name with image data

f = open(foggyfile,'r')
header = f.readline()

#loop through file and read data
a=[]
images = []; A0_indices = []; fog_indices = []; test_index=[]
for line in f: 	
	a=line.split()
	images.append(a[0])
	A0_indices.append(float(a[1]))
	fog_indices.append(float(a[2]))
	
f.close()

########## find good images and save composit ################
for i in range(len(A0_indices)):
	test_index.append(float(1-fog_indices[i])/float(1-A0_indices[i]))
   
test_index=np.array(test_index)
A0_indices = np.array(A0_indices)
fog_indices = np.array(fog_indices)

good = np.where( (A0_indices[:] >= A0_min[0])  &  (test_index[:] >= index_min[0]))
good = good[0]

# Create a numpy array of floats to store the average (assume RGB images)
w,h=Image.open(folder_in+images[0]).size
arr=np.zeros((h,w,3),np.float)

if len(good) > 5:
    print('Using best threshold')
    N=len(good)
    for i in range(len(good)):
        imarr=np.array(Image.open(folder_in+images[good[i]]),dtype=np.float)
        arr=arr+imarr/N
    
    # Round values in array and cast as 8-bit integer and save image
    arr=np.array(np.round(arr),dtype=np.uint8)
    out=Image.fromarray(arr,mode="RGB")
    out.save(folder_out+'final_'+todaystr+'.jpg')
else:
    good = np.where( (A0_indices[:] >= A0_min[1])  &  (test_index[:] >= index_min[1]))
    good=good[0]
    if len(good) > 5:
	print('Using fair threshold')
	N=len(good)
       	for i in range(len(good)):
             imarr=np.array(Image.open(folder_in+images[good[i]]),dtype=np.float)
             arr=arr+imarr/N
        
    	# Round values in array and cast as 8-bit integer and save image
    	arr=np.array(np.round(arr),dtype=np.uint8)
        out=Image.fromarray(arr,mode="RGB")
        out.save(folder_out+'final_'+todaystr+'.jpg')
