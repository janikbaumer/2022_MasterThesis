#!/usr/bin/env python

from matplotlib.image import imread
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
#todaystr = '20210105'

print('Processing images from ', todaystr)

images = []; A0_indices = []; fog_indices = []; timestamp = []

for filename in sorted(glob.glob('{}/{}*.jpg'.format( folder_in,todaystr ) )):
    img = imread(filename)/255.
    iu3 = int(np.shape(img)[0]/3)
    img_upper3 = img[:iu3,:,:]
    dl = np.min(img_upper3,axis=2)
    #print(iu3, np.shape(img), np.shape(img_upper3), np.shape(dl) )
    bl = np.max(img_upper3,axis=2)
    cl=bl-dl
    d=np.mean(dl)
    b=np.mean(bl)
    c=b-d
    A0 = np.max(bl)/3. +2*b/3.
    x1=(A0-d)/A0
    x2=c/A0
    fog_index=np.exp(-0.5*(5.1*x1+2.9*x2)+0.2461)

    imagename = filename.split('/')[-1] # filename without the foldername
    images.append(imagename)
    ts = datetime.datetime.strptime(imagename,'%Y%m%d%H%M%S.jpg') #Is it correct that images are saved every second????
    timestamp.append(ts)
    A0_indices.append(A0)
    fog_indices.append(fog_index)
    #print(filename.split('/')[-1])
    #for name,var in zip(['d','b','c','A0','x1','x2','fog_index'],[d,b,c,A0,x1,x2,fog_index]):
    #    print(name,'\t = ',var)
   
   
########## Save foggy file ################
foggyfile = '{}/{}.txt'.format(folder_out,todaystr)
f = open(foggyfile,'w')
f.write('#filename A0 fog_index\n')
for i in range(len(images)):
    f.write('{0} {1:.4f} {2:.4f}\n'.format(images[i],A0_indices[i],fog_indices[i])) #space seperated...{0} is imagename; {1:.4f} 1: means 2nd entry in format; .4f means float with 4 digits after comma   
f.close()
