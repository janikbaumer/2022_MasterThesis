import os
from os.path import isfile
import pandas as pd
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision import transforms
import numpy as np
from matplotlib import pyplot as plt


def get_indices_or_None(path, raw_img):
    """
    based on a given image name (.jpg),
    if corresponding txt file exists (that contains which imgs were used for composite image generation),
    then get corresponding A0 and fog index
    else (if txt file does not exist or has no entry for that day of input image)
    then return None
    """
    raw_img_jpg = raw_img[:-4]+'.jpg'  #  in case compressed (.png) files were used
    day = raw_img[0:8]  # get yyyymmdd from yyyymmddHHMMSS.jpg
    path_txtfile = os.path.join(path, f'{day}.txt')
    if not os.path.isfile(path_txtfile):  # file does not exist, set indices to None
        a_0, fog_index = None, None
    else:  # if txt file exists for that day
        with open(path_txtfile) as f:
            df = pd.read_csv(f, delim_whitespace=True, index_col='#filename')
            try:
                a_0 = df.loc[raw_img_jpg]['A0']
                fog_index = df.loc[raw_img_jpg]['fog_index']
            except KeyError:  # may happen if txt file exists, but does not contain indices for all images of that day
                a_0, fog_index = None, None
    return a_0, fog_index


def get_manual_label_or_None(path_to_file, file):
    """
    get the label from the text file with the manual classifications (from script handlabelling_raw.py)
    if file has not been labeled, then return None
    """
    full_path_manual_labels = os.path.join(path_to_file, 'manual_labels_isfoggy.txt')
    if not os.path.isfile(full_path_manual_labels):
        return None
    else:
        with open(full_path_manual_labels) as f:
            df = pd.read_csv(f, delim_whitespace=True, names=['filename', 'isfoggy'], index_col='filename')
            if file in df.index:
                label = df.at[file, 'isfoggy']
            else:
                label = None
        return label


def get_cam_thresholds(path):
    """
    get thresholds for given image from specific camera
    """
    path = os.path.join(path, 'thresholds.txt')
    with open(path) as f:
        df = pd.read_csv(f, delim_whitespace=True, names=['A0', 'fog_index'])

        A0_far0 = df.loc[0]['A0']
        fog_idx_far0 = df.loc[0]['fog_index']
        A0_optim = df.loc[1]['A0']
        fog_idx_optim = df.loc[1]['fog_index']
    return A0_optim, fog_idx_optim, A0_far0, fog_idx_far0


# same conditions as in matlab script (a06_*.m)
def check_thresholds(A0_img, fog_idx_img, A0_optim, fog_idx_optim, A0_far0, fog_idx_far0):
    """
    check whether thresholds were passed or not
    """
    idx_now = (1-fog_idx_img) / (1-A0_img)
    a0_pass = A0_img >= A0_far0
    fog_pass = idx_now >= fog_idx_far0
    return a0_pass, fog_pass


class DischmaSet_classification():

    def __init__(self, root='../datasets/dataset_downsampled/', stat_cam_lst=['Buelenberg_1'], mode='train') -> None:

        self.root = root
        self.stat_cam_lst = stat_cam_lst
        self.mode = mode
        
        self.DAYS_TRAIN = [str(ele).zfill(2) for ele in list(range(1, 24+1))]  # days 1 to 24 for training, residual days for validation
        self.DAYS_TEST = ['03', '10', '17', '25']

        # self.MONTHS_VAL = ['01', '04', '07', '10']  # use Jan, April, July, Oct for validation
        # self.MONTHS_TRAIN_VAL = ['01', '04', '07', '10']

        self.YEAR_TRAIN_VAL = '2021'
        self.YEAR_TEST = '2020'

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        self.denormalize = transforms.Compose(
            [transforms.Normalize(mean= [0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.Normalize(mean= [-0.485, -0.456, -0.406], std=[1., 1., 1.])]
        )  # ev not needed

        # rotation / affine transformations / random perspective probably make no sense (for one model per cam)
        # as camera installations will always be same (might make sense considering one model for multiple cameras)
        self.train_augmentation = transforms.RandomApply(torch.nn.ModuleList([
            transforms.RandomCrop(size=(int(0.8*400), int(0.8*600))),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.5, hue=0.3),  # as trees etc can change color over seasons
            self.normalize
        ]), p=1)

        self.val_test_augmentation = self.normalize

        # already done in preprocessing - manually saved and working with those
        self.DOWNSCALE_FACTOR = 1  # 1 for no downsampling, else height and width is (each) downscaled by this factor

        # create list of file names (raw images) for all stations / cams
        self.is_foggy = []
        self.path_list_valid = []
        
        for camstation in stat_cam_lst:
            STATION, CAM = camstation.split('_')

            self.PATH_RAW_IMAGE = os.path.join(root, 'DischmaCams', STATION, CAM)
            self.PATH_COMPOSITE = os.path.join(root, 'Composites', f'Cam{CAM}_{STATION}')
            self.full_path_manual_labels = os.path.join(self.PATH_RAW_IMAGE, 'manual_labels_isfoggy.txt')

            # get thresholds for this station/cam
            A0_optim, fog_idx_optim, A0_far0, fog_idx_far0 = get_cam_thresholds(self.PATH_COMPOSITE)

            # create list of filenames (raw images, sorted) (only of given years (2020, 2021))
            self.file_list_camstat_years = sorted([f for f in os.listdir(self.PATH_RAW_IMAGE) if (f.endswith('.jpg') or f.endswith('.png')) and isfile(os.path.join(self.PATH_RAW_IMAGE, f)) and (f[0:4] == self.YEAR_TEST or f[0:4] == self.YEAR_TRAIN_VAL)])

            # for each jpg/png file, get label from manual classification
            # create a label list of booleans (ordered the same as list with file names)
            for raw_img_name in self.file_list_camstat_years:  # raw_img_name: e.g. '20211230160501.jpg' or with .png
                full_path_img = os.path.join(self.PATH_RAW_IMAGE, raw_img_name)

                if self.mode == 'train' and raw_img_name[0:4] in self.YEAR_TRAIN_VAL and raw_img_name[6:8] in self.DAYS_TRAIN:
                    # training data is manually labelled from 2021 (from 1st to 24th)
                    if os.path.isfile(self.full_path_manual_labels):
                        img_is_foggy = get_manual_label_or_None(path_to_file=self.PATH_RAW_IMAGE, file=raw_img_name)
                        if img_is_foggy is not None:
                            self.is_foggy.append(int(img_is_foggy))
                            self.path_list_valid.append(full_path_img)

                if self.mode == 'val' and raw_img_name[0:4] in self.YEAR_TRAIN_VAL and raw_img_name[6:8] not in self.DAYS_TRAIN:
                    # validation data is manually labelled from 2021 (from 25th to end (28th/30th/31st))
                    # note: same code snippet as above
                    if os.path.isfile(self.full_path_manual_labels):
                        img_is_foggy = get_manual_label_or_None(path_to_file=self.PATH_RAW_IMAGE, file=raw_img_name)
                        if img_is_foggy is not None:
                            self.is_foggy.append(int(img_is_foggy))
                            self.path_list_valid.append(full_path_img)

                if self.mode == 'test' and raw_img_name[0:4] == self.YEAR_TEST and raw_img_name[6:8] in self.DAYS_TEST: # and raw_img_name[4:6] in self.MONTHS_TEST   # use manual labels
                    # testing data is manually labelled from 2020 (each 03rd, 10th, 17th and 25th from each month)
                    # checking that labels from SLF exist, then get manual classification (from txt file)
                    A0_img, fog_idx_img = get_indices_or_None(path=self.PATH_COMPOSITE, raw_img=raw_img_name)
                    if not (A0_img == None and fog_idx_img == None):  # if txt file for composite image generation exists and contains row with name of image)
                        if os.path.isfile(self.full_path_manual_labels):
                            img_is_foggy = get_manual_label_or_None(path_to_file=self.PATH_RAW_IMAGE, file=raw_img_name)
                            if img_is_foggy is not None:
                                self.is_foggy.append(int(img_is_foggy))
                                self.path_list_valid.append(full_path_img)

                if self.mode == 'baseline' and raw_img_name[0:4] == self.YEAR_TEST and raw_img_name[6:8] in self.DAYS_TEST:
                    # check that manual label exists (to get same as for test set), then get labels from SLF:
                        # for each jpg/png file, get indices, compare to camera thresholds
                        # and check whether img was used for composite image generation (if yes: not foggy (class 0) / if no: foggy (class 1))
                    if os.path.isfile(self.full_path_manual_labels):
                        img_is_foggy = get_manual_label_or_None(path_to_file=self.PATH_RAW_IMAGE, file=raw_img_name)
                        if img_is_foggy is not None:
                            
                            A0_img, fog_idx_img = get_indices_or_None(path=self.PATH_COMPOSITE, raw_img=raw_img_name)
                            if not (A0_img == None and fog_idx_img == None):  # if txt file for composite image generation exists and contains row with name of image)
                                a0_passed_th, fog_passed_th = check_thresholds(A0_img, fog_idx_img, A0_optim, fog_idx_optim, A0_far0, fog_idx_far0)
                                used_for_comp = a0_passed_th and fog_passed_th  # one bool, whether this img was used for composite image generation
                                img_is_foggy = not(used_for_comp)  # image is considered as foggy if it was not used for the composite image generation
                                self.is_foggy.append(int(img_is_foggy))
                                self.path_list_valid.append(full_path_img)  


    def __len__(self):
        """
        returns the number of VALID samples in our dataset
        """
        return len(self.path_list_valid)


    def __getitem__(self, idx):
        """
        returns a data sample from the dataset (this fct will be called <batch size> times, in every batch iteration)
        """

        # given idx, get image with filename belonging to this specific index
        self.path_img = self.path_list_valid[idx]

        # reading image
        image = torchvision.io.read_image(path=self.path_img)
        
        # downscaling image
        if self.DOWNSCALE_FACTOR != 1:
            shp = tuple(image.shape[-2:])  # (600, 400)
            shp_new = tuple(int(x/self.DOWNSCALE_FACTOR) for x in shp)
            image = F.resize(img=image, size=shp_new)
        
        # ensure shape is actually (400x600)
        if image.shape[1:] != torch.Size([400, 600]):
            image = F.resize(img=image, size=(400, 600))

        image = image/255  # convert to floats between 0 and 1  (normalization / standardization later)

        # transformations
        if self.mode == 'train':  # only do augmentation in training
            tf1 = self.train_augmentation(image)
        else:
            # tf1 = image
            tf1 = self.val_test_augmentation(image)  # also normalized data for validation / testing (/baseline)

        """
        # test plots:
        plt.imshow(np.transpose(image.numpy(), (1,2,0)))
        plt.imshow(np.transpose(self.denormalize(tf1).numpy(), (1,2,0))) # or without denormalize
        """

        # get label (True if img is foggy, resp. img was not used for composite image generation)
        label = self.is_foggy[idx]

        return tf1, label


    def get_balancedness(self):
        """
        returns number of images classified as foggy (1) and non-foggy (0)
        """
        n_clear = self.is_foggy.count(0)
        n_foggy = self.is_foggy.count(1)

        return n_clear, n_foggy


if __name__=='__main__':
    
    set1 = DischmaSet_classification(root='../datasets/dataset_downsampled/', stat_cam_lst=['Luksch_2'], mode='train')
    nclear, nfog = set1.get_balancedness()
    img, lbl = set1.__getitem__(0)
    
    print('nfog, nclear: ', nfog, nclear)
    print('total number of labeled images: ', set1.__len__())

    
    