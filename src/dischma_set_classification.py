import os
from os.path import isfile, join
import pandas as pd
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision import transforms
import numpy as np
from matplotlib import pyplot as plt


def get_indices_or_None(path, raw_img):
    """
    based on a given image name (.jpg), get corresponding A0 and fog index
    if corresponding txt file exists (that contains which imgs were used for composite image generation)
    else return None
    """
    raw_img_jpg = raw_img[:-4]+'.jpg'  #  in case compressed (.png) files were used
    day = raw_img[0:8]  # get yyyymmdd from yyyymmddHHMMSS.jpg
    path_txtfile = os.path.join(path, f'{day}.txt')
    if not os.path.isfile(path_txtfile):  # file does not exist
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
    get the label from the text file with the manual classifications (from script handlabelling.py)
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

    def __init__(self, root='../datasets/dataset_downsampled/', stat_cam_lst=['Buelenberg_1', 'Buelenberg_2'], mode='train') -> None:

        self.root = root
        self.stat_cam_lst = stat_cam_lst
        self.mode = mode
        self.YEAR_TRAIN = '2020'
        self.YEAR_VAL = '2021'
        self.MONTHS_VAL = ['01', '04', '07', '10']  # use Jan, April, July, Oct for validation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        self.denormalize = transforms.Compose(
            [transforms.Normalize(mean= [0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.Normalize(mean= [-0.485, -0.456, -0.406], std=[1., 1., 1.])]
        )

        self.train_augmentation = transforms.RandomApply(torch.nn.ModuleList([
            self.normalize,
            transforms.RandomCrop(size=(int(0.8*400), int(0.8*600))),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.GaussianBlur(kernel_size=5),
            # rotation / affine transformations / random perspective probably make no sense (for one model per cam), as camera installations will always be same (might make sense considering one model for multiple camera)
            transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.5, hue=0.3)  # might make sense (trees etc can change colors over seasons)
        ]), p=0.5)

        self.val_augmentation = self.normalize

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

            # create list of filenames (raw images, sorted) (only of given years)
            self.file_list_camstat_years = sorted([f for f in os.listdir(self.PATH_RAW_IMAGE) if (f.endswith('.jpg') or f.endswith('.png')) and isfile(os.path.join(self.PATH_RAW_IMAGE, f)) and (f[0:4] == self.YEAR_TRAIN or f[0:4] == self.YEAR_VAL)])

            # for each jpg/png file, get indices, compare to camera thresholds
            # and check whether img was used for composite image generation (if yes: not foggy (class 0) / if no: foggy (class 1))
            # create a label list of booleans (ordered the same as list with file names)

            for raw_img_name in self.file_list_camstat_years:  # raw_img_name: e.g. '20211230160501.jpg' or with .png
                full_path_img = os.path.join(self.PATH_RAW_IMAGE, raw_img_name)

                # TODO important !!!: create two new modes for getting data from only handlabelled images!
                if self.mode == 'train' and raw_img_name[0:4] == self.YEAR_TRAIN:  # use labels from txt files from PCA
                    A0_img, fog_idx_img = get_indices_or_None(path=self.PATH_COMPOSITE, raw_img=raw_img_name)
                    if not (A0_img == None and fog_idx_img == None):  # if txt file for composite image generation did not exist (or did not contain row with name of image)
                        a0_passed_th, fog_passed_th = check_thresholds(A0_img, fog_idx_img, A0_optim, fog_idx_optim, A0_far0, fog_idx_far0)
                        used_for_comp = a0_passed_th and fog_passed_th  # one bool, whether this img was used for composite image generation
                        img_is_foggy = not(used_for_comp)  # image is considered as foggy if it was not used for the composite image generation
                        self.is_foggy.append(int(img_is_foggy))
                        self.path_list_valid.append(full_path_img)

                if self.mode == 'train_manual' and raw_img_name[0:4] == self.YEAR_VAL and raw_img_name[4:6] not in self.MONTHS_VAL:
                    if os.path.isfile(self.full_path_manual_labels):
                        img_is_foggy = get_manual_label_or_None(path_to_file=self.PATH_RAW_IMAGE, file=raw_img_name)
                        if img_is_foggy is not None:
                            self.is_foggy.append(int(img_is_foggy))
                            self.path_list_valid.append(full_path_img)

                if self.mode == 'val' and raw_img_name[0:4] == self.YEAR_VAL and raw_img_name[4:6] in self.MONTHS_VAL:  # use manual labels
                    if os.path.isfile(self.full_path_manual_labels):
                        img_is_foggy = get_manual_label_or_None(path_to_file=self.PATH_RAW_IMAGE, file=raw_img_name)
                        if img_is_foggy is not None:
                            self.is_foggy.append(int(img_is_foggy))
                            self.path_list_valid.append(full_path_img)
                    else:
                        print('manual labels do not exist, make sure to label them first (create file manual_labels_isfoggy.txt (with script handlabelling.py!')
                        break

                """
                # ev usable for task above : adapt variable img_is_foggy with data from respective file 'manual_labels_isfoggy.txt'
                if mode == 'train' and raw_img_name[0:4] == self.YEAR_TRAIN:
                    self.is_foggy.append(int(img_is_foggy))
                    self.path_list_valid.append(full_path_img)
                elif mode == 'val' and raw_img_name[0:4] == self.YEAR_VAL and raw_img_name[4:6] in self.MONTHS_VAL:
                    self.is_foggy.append(int(img_is_foggy))
                    self.path_list_valid.append(full_path_img)
                """


    def __len__(self):
        """
        returns the number of VALID samples in our dataset
        """
        return len(self.path_list_valid)


    def __getitem__(self, idx):
        """
        returns a data sample from the dataset (this fct will be called <batch size> times, in every batch iteration)
        """

        # given idx, get image with filename belonging to this specific index (reading and downscaling)
        self.path_img = self.path_list_valid[idx]
        image = torchvision.io.read_image(path=self.path_img)
        if self.DOWNSCALE_FACTOR != 1:
            shp = tuple(image.shape[-2:])  # (600, 400)
            shp_new = tuple(int(x/self.DOWNSCALE_FACTOR) for x in shp)
            image = F.resize(img=image, size=shp_new)
        image = image/255  # convert to floats between 0 and 1  # normalization / standardization

        # transformations
        tf1 = self.train_augmentation(image)

        """
        # test plots:
        plt.imshow(np.transpose(image.numpy(), (1,2,0)))
        plt.imshow(np.transpose(self.denormalize(tf1).numpy(), (1,2,0))) # or without denormalize
        """

        # get label (True if img is foggy, resp. img was not used for composite image generation)
        label = self.is_foggy[idx]

        return tf1, label


    def get_balancedness(self):
        n_clear = self.is_foggy.count(0)
        n_foggy = self.is_foggy.count(1)  # nmbr of images classified as foggy

        return n_clear, n_foggy


if __name__=='__main__':
    x = DischmaSet_classification(root='../datasets/dataset_downsampled/', stat_cam_lst=['Stillberg_2'], mode='val')
    nclear, nfog = x.get_balancedness()
    img, lbl = x.__getitem__(0)
    print('nfog, nclear: ', nfog, nclear) 
    print('total number of labeled images: ', x.__len__())
