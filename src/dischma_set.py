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
        if len(ele) != 18 and ele[-1:-4] == '.jpg':  # 18: yyyymmddHHMMSS.jpg
            print('at least one file is not in correct format, should be: [yyyymmddHHMMSS].jpg')
            wrong_file = True
    if wrong_file == False:
        print('All files in correct format')
"""


# non optimal, txt files are read n_rows times (once for each raw image)
# based on a given image name (.jpg), get corresponding A0 and fog index
# if corresponding txt file exists (that notes which imgs were used for composite image generation)
# else return 0, 0
def get_indices(path, raw_img):
    raw_img_jpg = raw_img[:-4]+'.jpg'  #  in case compressed (.png) files were used
    day = raw_img[0:8]  # get yyyymmdd from yyyymmddHHMMSS.jpg
    path_txtfile = f'{path}/{day}.txt'
    if not os.path.isfile(path_txtfile):
        # TODO: go look in fog_index_all.mat file if there is an entry for this day
        a_0, fog_index = None, None
    else:  # if txt file exists for that day
        with open(path_txtfile) as f:
            df = pd.read_csv(f, delim_whitespace=True, index_col='#filename')
            try:  # TODO Mo, 28.02.2022 - do not check for whole image (contains also .jpg), but only check for substring (without extension!)
                a_0 = df.loc[raw_img_jpg]['A0']
                fog_index = df.loc[raw_img_jpg]['fog_index']
            except KeyError:  # may happen if txt file exists, but does not contain indices for all images of that day
                a_0, fog_index = None, None
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
    
    #####def __init__(self, root='../datasets/dataset_complete/', station='Buelenberg', camera='1') -> None:
    def __init__(self, root='../datasets/dataset_complete/', stat_cam_lst=['Buelenberg1', 'Buelenberg_2']):
        # create list of file names (raw images)
        # loop over all stations / cams (for devel, only choose BB1)
        # todo: when looping, consider that not all stations have 3 cameras
        # todo: when looping, maybe do nested list or change list name (depending on cam/station) -> consider that at same time, multiple images from diff cams exist
        
        self.DOWNSCALE_FACTOR = 1  # 1 for no downsampling, else height and width is (each) downscaled by this factor

        self.is_foggy = []
        self.file_list_valid = []

        for camstation in stat_cam_lst:
            STATION, CAM = camstation.split('_')

            self.PATH_RAW_IMAGE = root + f'DischmaCams/{STATION}/{CAM}'
            self.PATH_COMPOSITE = root + f'Composites/Cam{CAM}_{STATION}'
        
            # get thresholds for this station/cam
            A0_optim, fog_idx_optim, A0_far0, fog_idx_far0 = get_cam_thresholds(self.PATH_COMPOSITE)

            # create list of filenames (raw images, sorted)
            self.file_list_camstat = sorted([f for f in os.listdir(self.PATH_RAW_IMAGE) if (f.endswith('.jpg') or f.endswith('.png')) and isfile(os.path.join(self.PATH_RAW_IMAGE, f))])

            #check_files_second_format(file_list_camstat)


            # for each jpg/png file, get indices, compare to camera thresholds
            # and check whether img was used for composite image generation
            # create a label list of booleans (ordered the same as list with file names)

            # not done with dicts (keys cannot be accessed with indices (used for getitem method))
            for raw_img_name in self.file_list_camstat:  # raw_img_name: e.g. '20211230160501.jpg' or with .png
                A0_img, fog_idx_img = get_indices(path=self.PATH_COMPOSITE, raw_img=raw_img_name)
                if not (A0_img == None and fog_idx_img == None):  # if txt file for composite image generation did not exist (or did not contain row with name of image)
                    a0_passed_th, fog_passed_th = check_thresholds(A0_img, fog_idx_img, A0_optim, fog_idx_optim, A0_far0, fog_idx_far0)
                    used_for_comp = a0_passed_th and fog_passed_th  # one bool, whether this img was used for composite image generation
                    img_is_foggy = not(used_for_comp)  # image is considered as foggy if it was not used for the composite image generation
                    self.is_foggy.append(int(img_is_foggy))
                    self.file_list_valid.append(raw_img_name)
    
    def __len__(self):
        """
        returns the number of VALID samples in our dataset (where not both a0 and fog_idx are None)
        """
        return len(self.file_list_valid)


    def __getitem__(self, idx):
        """
        returns a data sample from the dataset (this fct will be called <batch size> times, in every loop)
        """

        # given idx, get image with filename belonging to this specific index
        self.img_name = self.file_list_valid[idx]
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
        # label = float(label)

        return image, label
    
    def get_balancedness(self):
        n_clear = self.is_foggy.count(0)
        n_foggy = self.is_foggy.count(1)  # nmbr of images classified as foggy

        return n_clear, n_foggy


if __name__=='__main__':
    # test what happens (call init function)
    x = DischmaSet(station='Stillberg', camera='3')
    nclear, nfog = x.get_balancedness()

    print('nfog, nclear: ', nfog, nclear)
    print('n total (labeled images): ', nfog + nclear)