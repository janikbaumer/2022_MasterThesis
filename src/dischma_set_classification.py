import os
from os.path import isfile, join
import pandas as pd
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision import transforms
import numpy as np
from matplotlib import pyplot as plt

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
    path_txtfile = os.path.join(path, f'{day}.txt')
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

def get_manual_label(path_to_file, file):
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
    idx_now = (1-fog_idx_img) / (1-A0_img)
    a0_pass = A0_img >= A0_far0
    fog_pass = idx_now >= fog_idx_far0
    return a0_pass, fog_pass


class DischmaSet_classification():

    ##### def __init__(self, root='../datasets/dataset_complete/', station='Buelenberg', camera='1') -> None:
    def __init__(self, root='../datasets/dataset_complete/', stat_cam_lst=['Buelenberg_1', 'Buelenberg_2'], mode='train'):
        # create list of file names (raw images)
        # loop over all stations / cams (for devel, only choose BB1)
        self.root = root
        self.stat_cam_lst = stat_cam_lst
        self.mode = mode
        self.YEAR_TRAIN = '2020'
        self.YEAR_VAL = '2021'
        self.MONTHS_VAL = ['01', '04', '07', '10']  # use Jan, April, July, Oct for validation


        self.train_augmentation = transforms.RandomApply(torch.nn.ModuleList([
            transforms.RandomCrop(size=(int(0.8*400), int(0.8*600))),
            transforms.RandomHorizontalFlip(p=0.5),  # here p=1, as p=0.5 will be applied for whole RandomApply block
            transforms.GaussianBlur(kernel_size=5),
            # rotation / affine transformations / random perspective probably make no sense (for one model per cam), as camera installations will always be same (might make sense considering one model for multiple camera)
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)  # might make sense (trees etc can change colors over seasons)
        ]), p=1)

        self.val_augmentation = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        self.normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        self.denormalize = transforms.Compose(
            [transforms.Normalize(mean= [0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.Normalize(mean= [-0.485, -0.456, -0.406], std=[1., 1., 1.])]
        )


        self.DOWNSCALE_FACTOR = 1  # 1 for no downsampling, else height and width is (each) downscaled by this factor

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

            #check_files_second_format(file_list_camstat_years)

            # for each jpg/png file, get indices, compare to camera thresholds
            # and check whether img was used for composite image generation
            # create a label list of booleans (ordered the same as list with file names)

            # not done with dicts (keys cannot be accessed with indices (used for getitem method))
            for raw_img_name in self.file_list_camstat_years:  # raw_img_name: e.g. '20211230160501.jpg' or with .png
                full_path_img = os.path.join(self.PATH_RAW_IMAGE, raw_img_name)

                if self.mode == 'train' and raw_img_name[0:4] == self.YEAR_TRAIN:  # use labels from txt files from PCA
                    A0_img, fog_idx_img = get_indices(path=self.PATH_COMPOSITE, raw_img=raw_img_name)
                    if not (A0_img == None and fog_idx_img == None):  # if txt file for composite image generation did not exist (or did not contain row with name of image)
                        a0_passed_th, fog_passed_th = check_thresholds(A0_img, fog_idx_img, A0_optim, fog_idx_optim, A0_far0, fog_idx_far0)
                        used_for_comp = a0_passed_th and fog_passed_th  # one bool, whether this img was used for composite image generation
                        img_is_foggy = not(used_for_comp)  # image is considered as foggy if it was not used for the composite image generation
                        self.is_foggy.append(int(img_is_foggy))
                        self.path_list_valid.append(full_path_img)
                
                if self.mode == 'train_manual' and raw_img_name[0:4] == self.YEAR_VAL and raw_img_name[4:6] not in self.MONTHS_VAL:
                    if os.path.isfile(self.full_path_manual_labels):
                        img_is_foggy = get_manual_label(path_to_file=self.PATH_RAW_IMAGE, file=raw_img_name)
                        if img_is_foggy is not None:
                            self.is_foggy.append(int(img_is_foggy))
                            self.path_list_valid.append(full_path_img)

                if self.mode == 'val' and raw_img_name[0:4] == self.YEAR_VAL and raw_img_name[4:6] in self.MONTHS_VAL:  # use manual labels
                    if os.path.isfile(self.full_path_manual_labels):
                        img_is_foggy = get_manual_label(path_to_file=self.PATH_RAW_IMAGE, file=raw_img_name)
                        if img_is_foggy is not None:
                            self.is_foggy.append(int(img_is_foggy))
                            self.path_list_valid.append(full_path_img)
                    else:
                        print('manual labels do not exist, make sure to label them first (create file manual_labels_isfoggy.txt (with script handlabelling.py!')
                        break
                """
                # TODO: adapt variable img_is_foggy with data from respective file 'manual_labels_isfoggy.txt'
                if mode == 'train' and raw_img_name[0:4] == self.YEAR_TRAIN:
                    self.is_foggy.append(int(img_is_foggy))
                    self.path_list_valid.append(full_path_img)
                elif mode == 'val' and raw_img_name[0:4] == self.YEAR_VAL and raw_img_name[4:6] in self.MONTHS_VAL:
                    self.is_foggy.append(int(img_is_foggy))
                    self.path_list_valid.append(full_path_img)
                """

            print()

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

        # self.img_name = self.path_list_valid[idx]
        self.path_img = self.path_list_valid[idx]


        #for camstation in self.stat_cam_lst:
        #    STATION, CAM = camstation.split('_')
        #    if os.path.isfile(f'{self.root}DischmaCams/{STATION}/{CAM}/{self.img_name}'):  # make sure to read files from correct directory - TODO: what happens when two images are taken at exactly the same second on different cams/stations?
        #        self.path_img = f'{self.root}{STATION}/{CAM}/{self.img_name}'  

        # self.path_img = f'{self.PATH_RAW_IMAGE}/{self.img_name}'

        # read and downscale image
        image = torchvision.io.read_image(path=self.path_img)
        shp = image.shape  # torch.Size([3, 6000, 4000])
        shp = tuple(image.shape[-2:])  # (6000, 4000)
        shp_new = tuple(int(x/self.DOWNSCALE_FACTOR) for x in shp)
        image = F.resize(img=image, size=shp_new)
        image = image.float()  # convert uint8 to float32

        # normalization / standardization
        image = image/255  # convert to values between 0 and 1
        
        """
        mean = image.mean((1,2))
        std = image.std((1,2))
        norm = torchvision.transforms.Normalize(mean, std)
        image = norm(image)
        """

        
        # transformations
        # plt.imshow(image.numpy().transpose(1,2,0))
        # plt.show()

        tf1 = self.train_augmentation(image)
        tf2 = self.normalize(tf1)
        #norm_image = self.transform_train(image)
        """
        # show unnormalized image:
        unnorm_img = self.inv_transform(norm_image)
        plt.imshow(np.transpose(unnorm_img.numpy(), (1, 2, 0)))
        """

        # get label (True if img is foggy, resp. img was not used for composite image generation)
        label = self.is_foggy[idx]
        # label = float(label)

        # return image, label
        return tf2, label


    def get_balancedness(self):
        n_clear = self.is_foggy.count(0)
        n_foggy = self.is_foggy.count(1)  # nmbr of images classified as foggy

        return n_clear, n_foggy


if __name__=='__main__':
    # test what happens (call init function)
    x = DischmaSet_classification(root='../datasets/dataset_downsampled/', stat_cam_lst=['Stillberg_2'], mode='val')
    nclear, nfog = x.get_balancedness()
    img, lbl = x.__getitem__(0)
    print('nfog, nclear: ', nfog, nclear)
    print('n total (labeled images): ', nfog + nclear)