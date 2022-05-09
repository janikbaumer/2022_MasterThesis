import os
from os.path import isfile, join
from re import fullmatch
from turtle import clear
import pandas as pd
from sklearn.semi_supervised import LabelSpreading
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

def ensure_same_days(imgs_paths, labels_paths):
    """
    only consider paths if image and label exist 
    lists will be sorted in same way (at same index, they must come from same day)
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

def get_full_resolution_label(image, label):
    """
    should return 4000, 6000 label, where missing values were filled up with 3 (no data),
    values are filled in at correct places (depending on affine transformation)
    """
    x_topleft, y_topleft = label.transform * (0, 0)  # to get x,y bottomright: label.transform * (label.width, label.height)
    x_shift, y_shift = int(abs(x_topleft)-0.5), int(abs(y_topleft)-0.5)  # -0.5 to get ints (no half numbers)

    full_shape = image.shape

    label_cropped = label.read()[0]

    # place label correctly in full sized image (4000, 6000)
    # initialize with all 3's, then replace the pixels where there is a label
    label_full = np.full(full_shape, 3)
    label_full[y_shift:y_shift+label_cropped.shape[0], x_shift:x_shift+label_cropped.shape[1]] = label_cropped  # get full label 

    label_full = label_full[np.newaxis, :]

    """
    nbr = 0
    p = f'correctly_placed_labels/number_{nbr}.png'
    while os.path.isfile(p):
        nbr += 1
        p = f'correctly_placed_labels/number_{nbr}.png'
    plt.imsave(fname=p, arr=label_full)
    """

    return label_full

def plot_array(img):
    plt.imshow(np.transpose(img, (1,2,0)))


'''
def plot_overlay(x, y):
    """
    no data where y==3
    snow/no snow y==1/0

    create boolean mask with
        true if snow/no snow
        false if no data
    apply mask to image (be aware that image is 3D)
    TODO: in case (later) x, y are not (anymore) numpy, first convert 
    """

    # TODO: ADD mask with shape of x (not of y!)  --> fill up residual pixels with 3 (resp, false)
    # or maybe dont do line above, but make sure that y comes as array of x.shape -> __getitem__()
        # make sure to consider affine transformation / transform correctly !!!
    mask = (y!= 3)
    mask_sq = mask.squeeze()
    full_mask = np.stack((mask_sq, mask_sq, mask_sq))  # to use for 3D image


    
    # Use the syntax array[array_condition] = value to replace each element in array with value if it meets the array_condition.
    

    x[full_mask] = y[mask]

    print()
'''

class DischmaSet_segmentation():

    def __init__(self, root='../datasets/dataset_downsampled/', stat_cam_lst=['Buelenberg_1', 'Buelenberg_2'], mode='train') -> None:
        """
        get lists with filenames from corresponding input cams
        note: working with downsampled images (for now)
        some definitions of variables (inputs)
        two lists:
            - one contains filepaths with composite images (downsampled)
            - the other contains filepaths with labels (from tif files) (downsampled)
            (for all imgs in camstat list -> loop!)
        TODO: define data augmentation pipeline
        """
        self.root = root
        self.stat_cam_lst = stat_cam_lst
        self.mode = mode

        self.patch_size = (400, 600)
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

        ### calc offset for each camstat 
        # pixels needed to move from topleft corner - are always the same per camstat (from certain date, see excel file from slack: 'Zeitpunkte_Georef_DischmaCams.xlsx')
        # tested few times visually and acc to ehafner
        """
        for camstation in stat_cam_lst:
            pass
        """
        
    def __len__(self):
        """
        returns the number of VALID samples in dataset
        """
        return len(self.label_path_list)

    def __getitem__(self, idx):
        """
        TODO:
        for given idx, choose corresponding img / label (see dischma_set_classification)
        """
        img_path = self.compositeimage_path_list[idx]
        label_path = self.label_path_list[idx]

        image = rasterio.open(img_path)
        label = rasterio.open(label_path)

        # consider windowed reading and writing: https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html#windowrw
        # problematic with getting full resolution of label while considering metadata and knowing exactly which patch
        # ev possible, but computationally expensive

        img = image.read()/255.  # should return 4000, 6000 image, vals between 0 and 1
        lbl = get_full_resolution_label(image, label)


        # TODO: only consider ceratin patch 
        #random_offset_width = random.randint(0, image.width-self.patch_size[0])
        #random_offset_height = random.randint(0, image.height-self.patch_size[1])

        # ev do some data augmentation


        # assert divisible by 32 (assume here img still has shape 4000, 6000)
        img = crop_center(img, img.shape[1], int(img.shape[2]-32/2))
        lbl = crop_center(lbl, lbl.shape[1], int(lbl.shape[2]-32/2))

        img = crop_random_patch(img, self.patch_size)
        lbl = crop_random_patch(lbl, self.patch_size)

        
        return img, lbl

def crop_center(arr, cropy, cropx):
    ch, y,x = arr.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return arr[:, starty:starty+cropy, startx:startx+cropx]

def crop_random_patch(array, patch_size):
    y_max = array.shape[1]
    x_max = array.shape[2]

    x_step = patch_size[0]
    y_step = patch_size[1]
    
    x_start = random.randrange(x_max-x_step)
    y_start = random.randrange(y_max-y_step)

    array_patch = array[:, y_start:y_start+y_step, x_start:x_start+x_step]

    return array_patch


if __name__=='__main__':
    # test what happens (call init function)
    all = ['Buelenberg_1', 'Buelenberg_2', 'Giementaelli_1', 'Giementaelli_2', 'Giementaelli_3', 'Luksch_1', 'Luksch_2', 'Sattel_1', 'Sattel_2', 'Sattel_3', 'Stillberg_1', 'Stillberg_2', 'Stillberg_3']
    some = ['Sattel_1', 'Stillberg_1', 'Stillberg_2', 'Buelenberg_1']

    x = DischmaSet_segmentation(root='../datasets/dataset_complete/', stat_cam_lst=all, mode='val')
    img, lbl = x.__getitem__(11)

    # img = img/255.

    # to overlay images
    # convert to PIL Images
    i = Image.fromarray(np.transpose(np.uint8(img*255), (1,2,0)))
    l = Image.fromarray(np.uint8(np.squeeze(lbl, axis=0)))
    # https://de.acervolima.com/uberlagern-sie-ein-bild-mit-einem-anderen-bild-in-python/
    i.paste(l, (0,0), mask=l)
    # i.show()

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
