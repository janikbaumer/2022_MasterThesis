import os
from os.path import isfile, join
import pandas as pd
import torch
import torchvision
from torchvision.transforms import functional as F

"""
def check_files_second_format(lst):
    wrong_file = False
    for ele in lst:
        # check that all files are in format [yyyymmddHHMMSS], not [yyyymmdd]
        if len(ele) != 18 and ele[-1:-4] == '.jpgd':  # 18: yyyymmddHHMMSS.jpg
            print('at least one file is not in correct format, should be: [yyyymmddHHMMSS].jpg')
            wrong_file = True
    if wrong_file == False:
        print('All files in correct format')
"""


# non optimal, txt files are read n_rows times (once for each raw image)
# based on a given image name (.jpg), get corresponding A0 and fog index
def get_indices(path, raw_img):
    day = raw_img[0:8]  # get yyyymmdd from yyyymmddHHMMSS.jpg
    path_txtfile = f'{path}/{day}.txt'
    if not os.path.isfile(path_txtfile):
        # todo: go look in fog_index_all.mat file if there is an entry for this day
        print('txt file for this day does not exist - TODO (skip this day!!!)')
    else:  # if txt file exists for that day
        with open(path_txtfile) as f:
            df = pd.read_csv(f, delim_whitespace=True, index_col='#filename')
            a_0 = df.loc[raw_img]['A0']
            fog_index = df.loc[raw_img]['fog_index']
    return a_0, fog_index

def get_cam_thresholds(path):
    path = f'{path}/thresholds.txt'
    with open(path) as f:
        df = pd.read_csv(f, delim_whitespace=True, names=['A0', 'fog_index'])
        
        A0_far0 = df.loc[0]['A0']
        fog_idx_far0 = df.loc[0]['fog_index']
        A0_optim = df.loc[1]['A0']
        fog_idx_optim = df.loc[1]['fog_index']
    return A0_optim, fog_idx_optim, A0_far0, fog_idx_far0

# same conditions as in matlab script (a06_*.m)
def check_thresholds(A0_img, fog_idx_img, A0_optim, fog_idx_optim, A0_far0, fog_idx_far0):
    idx_now = (1-fog_idx_img) / (1-A0_img)
    a0_pass = A0_img >= A0_far0
    fog_pass = idx_now >= fog_idx_far0
    return a0_pass, fog_pass


class DischmaSet():
    
    def __init__(self, root='../datasets/dataset_devel/') -> None:
        
        # create list of file names (raw images)
        # loop over all stations / cams (for devel, only choose BB1)
        # todo: when looping, consider that not all stations have 3 cameras
        # todo: when looping, maybe do nested list or change list name (depending on cam/station) -> consider that at same time, multiple images from diff cams exist
        
        STATION = 'Buelenberg'
        CAM = '1'

        self.DOWNSCALE_FACTOR = 10  # 1 for no downsampling, else height and width is (each) downscaled by this factor

        self.PATH_RAW_IMAGE = root + f'DischmaCams/{STATION}/{CAM}'
        self.PATH_COMPOSITE = root + f'Composites/Cam{CAM}_{STATION}'
        
        # get thresholds for this station/cam
        A0_optim, fog_idx_optim, A0_far0, fog_idx_far0 = get_cam_thresholds(self.PATH_COMPOSITE)

        # create list of filenames (raw images, sorted)
        self.file_list = sorted([f for f in os.listdir(self.PATH_RAW_IMAGE) if isfile(os.path.join(self.PATH_RAW_IMAGE, f))])
        self.img_used_for_comp_list = []
        self.is_foggy = []
        #check_files_second_format(file_list)

        # for each jpg file, get indices, compare to camera thresholds
        # and check whether img was used for composite image generation
        # create a label list of booleans (ordered the same as list with file names)

        # not done with dicts (keys cannot be accessed with indices (used for getitem method))

        for raw_img_name in self.file_list:
            A0_img, fog_idx_img = get_indices(path=self.PATH_COMPOSITE, raw_img=raw_img_name)
            a0_passed_th, fog_passed_th = check_thresholds(A0_img, fog_idx_img, A0_optim, fog_idx_optim, A0_far0, fog_idx_far0)
            used_for_comp = a0_passed_th and fog_passed_th  # one bool, whether this img was used for composite image generation
            img_is_foggy = not(used_for_comp)  # image is considered as foggy if it was not used for the composite image generation
            self.is_foggy.append(int(img_is_foggy))
        print('lists have same length? (should be T): ', len(self.file_list) == len(self.is_foggy))
    

    def __len__(self):
        """
        returns the number of samples in our dataset
        """
        return len(self.file_list)


    def __getitem__(self, idx):
        """
        returns a data sample from the dataset (this fct will be called <batch size> times)
        """

        # given idx, get image with filename belonging to this specific index
        self.img_name = self.file_list[idx]
        self.path_img = f'{self.PATH_RAW_IMAGE}/{self.img_name}'
        
        # read and downscale image
        image = torchvision.io.read_image(path=self.path_img)
        shp = image.shape  # torch.Size([3, 6000, 4000])
        shp = tuple(image.shape[-2:])  # (6000, 4000)
        shp_new = tuple(int(x/self.DOWNSCALE_FACTOR) for x in shp)
        image = F.resize(img=image, size=shp_new)
        image = image.float()  # convert uint8 to float32
        
        # get label (True if img is foggy, resp. img was not used for composite image generation)
        label = self.is_foggy[idx]

        return image, label


# test what happens (call init function)
x = DischmaSet()

