import os
from os.path import isfile, join
import pandas as pd
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision import transforms
import numpy as np
from matplotlib import pyplot as plt
import rasterio
from rasterio.windows import Window
from rasterio.plot import show
import rasterio
from rasterio.plot import show
from PIL import Image
import random


############ REPRODUCIBILITY ############

# Fix all random seeds
random_seed = 42
torch.seed()  # only for CPU
torch.manual_seed(random_seed)  # should work for CPU and CUDA
random.seed(random_seed)
np.random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(random_seed)

def get_label_patch_and_shape_and_tf(lbl_path, patchsize, x_rand=3000, y_rand=2000):
    x_patch = patchsize[0]
    y_patch = patchsize[1]

    with rasterio.open(lbl_path) as src:
        full_shape = src.shape
        arr = src.read(window=Window(x_rand, y_rand, x_patch, y_patch))
        x_topleft, y_topleft = src.transform * (0, 0)  # to get x,y bottomright: label.transform * (label.width, label.height)
        x_shift, y_shift = int(abs(x_topleft)-0.5), int(abs(y_topleft)-0.5)  # -0.5 to get ints (no half numbers)

    return arr, full_shape, x_shift, y_shift

def get_image_patch(img_path, x_shift, y_shift, patchsize, x_rand=600, y_rand=400):
    x_patch = patchsize[0]
    y_patch = patchsize[1]

    with rasterio.open(img_path) as src:
        arr = src.read(window=Window(col_off=x_rand+x_shift, row_off=y_rand+y_shift, width=x_patch, height=y_patch))

    return arr

def ensure_same_days(imgs_paths, labels_paths):
    """
    only consider paths if image and label exist 
    both lists will be sorted in same way (at same index, the paths must be from same day)
    """
    imgs_path_new = []
    labels_path_new = []

    # two dicts mapping from STATION_CAM_DATE to FULL_PATH (to label, resp. to composite image)
    img_specs = {}
    for img_path in imgs_paths:
        img_path_wo_file, img_file = os.path.split(img_path)
        _, img_camstat = os.path.split(img_path_wo_file)  # camstat: 'CamX_STATION'
        img_station = img_camstat[5:]
        img_cam = img_camstat[3]
        img_day = img_file[6:14]
        img_specs[f'{img_station}_{img_cam}_{img_day}'] = img_path

    label_specs = {}
    for label_path in labels_paths:
        label_path_wo_file, label_file = os.path.split(label_path)
        _, label_camstat = os.path.split(label_path_wo_file)  # camstat: 'CamX_STATION'
        label_camstat = label_camstat[14:]

        label_station = label_camstat.split('_')[0]
        label_cam = label_camstat.split('_')[1][-1]
        label_day = label_file[6:14]
        label_specs[f'{label_station}_{label_cam}_{label_day}'] = label_path

    # intersection
    both_available = [val for val in img_specs.keys() if val in label_specs.keys()]
    for key in img_specs:
        if key in both_available:
            imgs_path_new.append(img_specs[key])

    for key in label_specs:
        if key in both_available:
            labels_path_new.append(label_specs[key])

    return imgs_path_new, labels_path_new

"""
nbr = 0
p = f'correctly_placed_labels/number_{nbr}.png'
while os.path.isfile(p):
    nbr += 1
    p = f'correctly_placed_labels/number_{nbr}.png'
plt.imsave(fname=p, arr=label_full)
"""


class DischmaSet_segmentation():

    def __init__(self, root='../datasets/dataset_downsampled/', stat_cam_lst=['Buelenberg_1', 'Buelenberg_2'], mode='train') -> None:
        """
        some definitions of variables (inputs)
        get lists with filenames from corresponding input cams
        two lists:
            - one contains filepaths with composite images
            - the other contains filepaths with labels from tif files

        TODO: define data augmentation pipeline
        """
        self.root = root
        self.stat_cam_lst = stat_cam_lst
        self.mode = mode

        self.original_shape = (4000, 6000)  # shape of images - hardcoded, so image metadata not needed to read everytime
        self.patch_size = (256, 256)
        if self.patch_size[0]%32 != 0 or self.patch_size[1]%32 != 0: # for Unet, make sure both are divisible by 32 !!!
            print('Warning: patch size must be divisible by 32 in both dimensions !')
            print('check variable self.patch_size (DischmaSet_segmentation.__init__()')

        ### get lists with paths from composite images and corrsponding labels
        self.compositeimage_path_list = []
        self.label_path_list = []
        for camstation in stat_cam_lst:
            STATION, CAM = camstation.split('_')
            print(STATION, CAM)

            self.PATH_COMP_IMG = os.path.join(self.root, f'Composites/Cam{CAM}_{STATION}')
            self.file_list_camstat_imgs = sorted([f for f in os.listdir(self.PATH_COMP_IMG) if (f.endswith('.png') or f.endswith('.jpg')) and isfile(os.path.join(self.PATH_COMP_IMG, f))]) 
            for file in self.file_list_camstat_imgs:
                self.compositeimage_path_list.append(os.path.join(self.PATH_COMP_IMG, file))

            self.PATH_LABELS = os.path.join(self.root, f'final_workdir_{STATION}_Cam{CAM}')
            self.file_list_camstat_lbl = sorted([f for f in os.listdir(self.PATH_LABELS) if f.endswith('.tif') and isfile(os.path.join(self.PATH_LABELS, f))])
            for file in self.file_list_camstat_lbl:
                self.label_path_list.append(os.path.join(self.PATH_LABELS, file))
        self.compositeimage_path_list, self.label_path_list = ensure_same_days(self.compositeimage_path_list, self.label_path_list)

    def __len__(self):
        """
        returns the number of VALID samples in dataset
        """
        return len(self.label_path_list)

    def __getitem__(self, idx):
        """
        return small patches from full images / labels for patch wise processing
        only patches are read (not full images) - faster processing
        """
        img_path = self.compositeimage_path_list[idx]
        label_path = self.label_path_list[idx]

        # TODO:
        # normalize image
        # data augmentation 

        label_shape = rasterio.open(label_path).shape
        xrand = random.randint(0, label_shape[1] - self.patch_size[1])
        yrand = random.randint(0, label_shape[0] - self.patch_size[0])

        # get image and label patches
        lbl_patch, lbl_shape_full, xshift, yshift = get_label_patch_and_shape_and_tf(label_path, self.patch_size, x_rand=xrand, y_rand=yrand)
        img_patch = get_image_patch(img_path, xshift, yshift, self.patch_size, x_rand=xrand, y_rand=yrand)

        # img = image.read()/255.  # should return 4000, 6000 image, vals between 0 and 1
        # ev do some data augmentation

        '''
        plt.figure()
        f, axarr = plt.subplots(2, 1) #subplot(r,c) provide the no. of rows and columns
        axarr[0].set_title(f'Image Patch {self.patch_size}')
        axarr[0].imshow(np.transpose(img_patch, (1,2,0)))
        axarr[1].set_title(f'Label Patch {self.patch_size}')
        axarr[1].imshow(np.transpose(lbl_patch, (1,2,0)))

        plt.figure()
        f, axarr = plt.subplots(2, 1) #subplot(r,c) provide the no. of rows and columns
        axarr[0].set_title(f'Full Image {self.original_shape}')
        axarr[0].imshow(np.transpose(i, (1,2,0)))
        axarr[1].set_title(f'Full Label {lbl_shape_full}')
        axarr[1].imshow(np.transpose(l, (1,2,0)))
        print()
        '''

        return img_patch, lbl_patch


if __name__=='__main__':
    # testing functionalities

    all = ['Buelenberg_1', 'Buelenberg_2', 'Giementaelli_1', 'Giementaelli_2', 'Giementaelli_3', 'Luksch_1', 'Luksch_2', 'Sattel_1', 'Sattel_2', 'Sattel_3', 'Stillberg_1', 'Stillberg_2', 'Stillberg_3']
    some = ['Sattel_1', 'Stillberg_1', 'Stillberg_2', 'Buelenberg_1']

    x = DischmaSet_segmentation(root='../datasets/dataset_complete/', stat_cam_lst=all, mode='val')
    img, lbl = x.__getitem__(11)

    """
    # to overlay images
    # convert to PIL Images
    i = Image.fromarray(np.transpose(np.uint8(img*255), (1,2,0)))
    l = Image.fromarray(np.uint8(np.squeeze(lbl, axis=0)))
    # https://de.acervolima.com/uberlagern-sie-ein-bild-mit-einem-anderen-bild-in-python/
    i.paste(l, (0,0), mask=l)
    # i.show()
    """

    """
    x = DischmaSet_segmentation(root='../datasets/dataset_complete/', stat_cam_lst=all[0:], mode='val')
    img, lbl = x.__getitem__(10)
    x = DischmaSet_segmentation(root='../datasets/dataset_complete/', stat_cam_lst=all[1:], mode='val')
    img, lbl = x.__getitem__(10)
    x = DischmaSet_segmentation(root='../datasets/dataset_complete/', stat_cam_lst=all[2:], mode='val')
    img, lbl = x.__getitem__(10)
    x = DischmaSet_segmentation(root='../datasets/dataset_complete/', stat_cam_lst=all[3:], mode='val')
    img, lbl = x.__getitem__(10)
    x = DischmaSet_segmentation(root='../datasets/dataset_complete/', stat_cam_lst=all[4:], mode='val')
    img, lbl = x.__getitem__(10)
    x = DischmaSet_segmentation(root='../datasets/dataset_complete/', stat_cam_lst=all[5:], mode='val')
    img, lbl = x.__getitem__(10)
    x = DischmaSet_segmentation(root='../datasets/dataset_complete/', stat_cam_lst=all[6:], mode='val')
    img, lbl = x.__getitem__(10)
    x = DischmaSet_segmentation(root='../datasets/dataset_complete/', stat_cam_lst=all[7:], mode='val')
    img, lbl = x.__getitem__(10)
    x = DischmaSet_segmentation(root='../datasets/dataset_complete/', stat_cam_lst=all[8:], mode='val')
    img, lbl = x.__getitem__(10)
    x = DischmaSet_segmentation(root='../datasets/dataset_complete/', stat_cam_lst=all[9:], mode='val')
    img, lbl = x.__getitem__(10)
    x = DischmaSet_segmentation(root='../datasets/dataset_complete/', stat_cam_lst=all[10:], mode='val')
    img, lbl = x.__getitem__(10)
    x = DischmaSet_segmentation(root='../datasets/dataset_complete/', stat_cam_lst=all[11:], mode='val')
    img, lbl = x.__getitem__(10)
    x = DischmaSet_segmentation(root='../datasets/dataset_complete/', stat_cam_lst=all[12:], mode='val')
    img, lbl = x.__getitem__(10)
    """
