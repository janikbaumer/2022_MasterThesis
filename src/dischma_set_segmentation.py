from cmath import phase
import os
import pandas as pd
import torch
import numpy as np
import rasterio
import rasterio
import random
import warnings
import rasterio

from torchvision import transforms
from matplotlib import pyplot as plt
from rasterio.plot import show
from rasterio.windows import Window
from os.path import isfile

from dischma_set_classification import get_manual_label_or_None

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
print('imports done')

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


############ FUNCTIONS ############

def get_label_patch_and_tf(lbl_path, patchsize, x_rand, y_rand):
    x_patch = patchsize[0]
    y_patch = patchsize[1]

    with rasterio.open(lbl_path) as src:
        arr = src.read(window=Window(x_rand, y_rand, x_patch, y_patch))
        x_topleft, y_topleft = src.transform * (0, 0)  # to get x,y bottomright: label.transform * (label.width, label.height)
        x_shift, y_shift = int(abs(x_topleft)-0.5), int(abs(y_topleft)-0.5)  # -0.5 to get ints (no half numbers)
    arr[arr==85] = 1  # needed?
    arr[arr==255] = 2  # needed?
    arr[arr==3] = 2

    return arr, x_shift, y_shift


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

def get_nonfoggy(lst, path):
    nonfoggy = []
    for ele in lst:
        img = ele.split('/')[-1]
        img_is_foggy = get_manual_label_or_None(path_to_file=path, file=img, task='segmentation')
        if img_is_foggy == 0:
            nonfoggy.append(ele)

    return nonfoggy
    # append non foggy images to list

def data_augmentation(tensor_img_lbl, augm_pipeline, gblur, norm, coljit):
    imglbl = augm_pipeline(tensor_img_lbl)  # together to have same random seed (horizontal flip (+ev others) with same probability)
    img = imglbl[0:3, ...]
    img = gblur(img)
    img = coljit(img)
    img = img.to(dtype=torch.float)
    img = norm(img)
    

    lbl = imglbl[-1, ...].unsqueeze(0)
    # lbl = norm(lbl)
    # lbl = coljit(lbl)

    return img.to(torch.float64), lbl.long()  # augmented_imglbl


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
    # initialize with all 0's, then replace the pixels where there is a label
    label_full = np.full(full_shape, 0)
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




class DischmaSet_segmentation():

    def __init__(self, root='../datasets/dataset_complete/', stat_cam_lst=['Buelenberg_1'], mode='train') -> None:
        """
        get lists with filenames from corresponding input cams
        two lists:
            - one contains filepaths with composite images
            - the other contains filepaths with labels from tif files (final_workdir_...)
        """

        self.root = root
        self.stat_cam_lst = stat_cam_lst
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.original_shape = (4000, 6000)  # shape of images - hardcoded, so image metadata not needed to be read everytime
        self.patch_size = (256*4, 256*4)  # (256*8, 256*8) for clearly visible images
        if self.patch_size[0]%32 != 0 or self.patch_size[1]%32 != 0: # for Unet, make sure both dims are divisible by 32 !!!
            print('Warning: patch size must be divisible by 32 in both dimensions !')
            print('check variable self.patch_size (DischmaSet_segmentation.__init__()')

        self.DAYS_TRAIN = [str(ele).zfill(2) for ele in list(range(1, 24+1))]  # days 1 to 24 for training, residual days for validation
        self.DAYS_TEST = ['03', '10', '17', '25']

        self.YEAR_TRAIN_VAL = '2021'
        self.YEAR_TEST = '2020'

        self.MT_DAYS_TEST = {
            'Buelenberg_1' : ['0116', '0416', '0715', '1017'],  # 16th of Jan, 16th of April, 15th of July etc
            'Buelenberg_2' : ['0116', '0416', '0712', '1014'],
            'Giementaelli_1' : ['0116', '0417', '0718', '1017'],  # TODO: check what ehafner wrote, ev one img in Gt1 and Gt2 are mixed (wrong directory)
            'Giementaelli_2' : ['0116', '0416', '0801', '1017'],  # TODO same as above
            'Giementaelli_3' : ['0116', '0416', '0713', '1017'],  # TODO: some were Luksch_2, not Giementaelli_3
            'Luksch_1': ['0116', '0417', '0624', '0914'],
            'Luksch_2': ['0116', '0417', '0820', '1018'],
            'Sattel_1': ['0116', '0416', '0625', '1017'],
            'Sattel_2': ['0117', '0416', '0625', '1017'],
            'Sattel_3': ['0116', '0416', '0625', '0908'],
            'Stillberg_1': ['0116', '0416', '0806', '1014'],
            'Stillberg_2': ['0116', '0416', '0521', '0905'],
            'Stillberg_3': ['0116', '0416', '0521', '0905']
        }

        # data augmentation
        mean_ImageNet = np.asarray([0.485, 0.456, 0.406])
        std_ImageNet = np.asarray([0.229, 0.224, 0.225])

        self.normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.denormalize = transforms.Normalize((-1*mean_ImageNet/std_ImageNet), (1.0/std_ImageNet))

        self.train_augmentation = transforms.RandomApply(torch.nn.ModuleList([
            # transforms.RandomCrop(size=(int(0.8*self.patch_size[0]), int(0.8*self.patch_size[1]))),  # already cropped when choosing patches
            transforms.RandomHorizontalFlip(p=1),  # here p=1, as p=0.5 will be applied for whole RandomApply block
            # rotation / affine transformations / random perspective probably make no sense (for one model per cam), as camera installations will always be same (might make sense considering one model for multiple camera)
            # transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.5, hue=0.3)  # might make sense (trees etc can change colors over seasons) - uncomment to check images (for manual segmentation)
            ]), p=0.5)
        self.gblur = transforms.GaussianBlur(kernel_size=5)
        self.coljit = transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.5, hue=0.3)


        # get lists with paths from composite images and corrsponding labels
        self.compositeimage_path_list = []
        self.label_path_list = []
        for camstation in stat_cam_lst:
            STATION, CAM = camstation.split('_')

            self.PATH_COMP_IMG = os.path.join(self.root, f'Composites/Cam{CAM}_{STATION}')
            self.file_list_camstat_imgs = sorted([f for f in os.listdir(self.PATH_COMP_IMG) if (f.endswith('.png') or f.endswith('.jpg')) and isfile(os.path.join(self.PATH_COMP_IMG, f))]) 
            for file in self.file_list_camstat_imgs:
                yr = file[6:10]
                day = file[12:14]
                mt_day = file[10:14]

                if mode == 'train':
                    if day in self.DAYS_TRAIN and yr in self.YEAR_TRAIN_VAL:
                        self.compositeimage_path_list.append(os.path.join(self.PATH_COMP_IMG, file))
                elif mode == 'val':
                    if day not in self.DAYS_TRAIN and yr in self.YEAR_TRAIN_VAL:
                        self.compositeimage_path_list.append(os.path.join(self.PATH_COMP_IMG, file))
                elif mode == 'test' or mode == 'baseline':  # same composites for testing and baseline (but different labels, see below)
                    if mt_day in self.MT_DAYS_TEST[camstation] and yr in self.YEAR_TEST:
                        self.compositeimage_path_list.append(os.path.join(self.PATH_COMP_IMG, file))

            # remove foggy images (in case composite image is foggy)
            # this is checked manually for test set, otherwise iso segmentation is not done ! (same valid for baseline)
            if self.mode != 'test' and self.mode != 'baseline':
                self.compositeimage_path_list = get_nonfoggy(lst=self.compositeimage_path_list, path=self.PATH_COMP_IMG)


            self.PATH_LABELS = os.path.join(self.root, f'final_workdir_{STATION}_Cam{CAM}')
            self.file_list_camstat_lbl = sorted([f for f in os.listdir(self.PATH_LABELS) if f.endswith('.tif') and isfile(os.path.join(self.PATH_LABELS, f))])
            for file in self.file_list_camstat_lbl:
                yr = file[6:10]
                day = file[12:14]
                mt_day = file[10:14]

                if mode == 'train':
                    if day in self.DAYS_TRAIN and yr in self.YEAR_TRAIN_VAL:
                        self.label_path_list.append(os.path.join(self.PATH_LABELS, file))
                elif mode == 'val':
                    if day not in self.DAYS_TRAIN and yr in self.YEAR_TRAIN_VAL:
                        self.label_path_list.append(os.path.join(self.PATH_LABELS, file))
                elif mode == 'test':
                    if mt_day in self.MT_DAYS_TEST[camstation] and yr in self.YEAR_TEST and file.endswith('_iso_simplified_clip.tif'):
                        self.label_path_list.append(os.path.join(self.PATH_LABELS, file))
                elif mode == 'baseline':  # take labels from SLF (old), not new ones with iso clustering
                    if mt_day in self.MT_DAYS_TEST[camstation] and yr in self.YEAR_TEST and not file.endswith('_iso_simplified_clip.tif'):  # TODO: check if condition is correct
                        self.label_path_list.append(os.path.join(self.PATH_LABELS, file))

                # note: training and validation are with labels that were used by SLF so far
                # note: baseline are the labels that were used by SLF so far (but same images as for test set)
                # note: test set are images with unsupervised iso clustering and manual enhancement

        self.compositeimage_path_list, self.label_path_list = ensure_same_days(self.compositeimage_path_list, self.label_path_list)

        # TODO: make sure to all days in compositeimage_path_list, resp. label_path_list a coresponding baseline exists !!!


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

        if self.mode == 'test' or self.mode == 'baseline':
            # get image and label from manual segmentation

            #img_path = '/scratch2/jbaumer/2022_MasterThesis/datasets/dataset_testset/BB1/20200116/final_20200116.jpg'
            #lbl_path_normal = '/scratch2/jbaumer/2022_MasterThesis/datasets/dataset_testset/BB1/20200116/final_20200116_extract_snowclass_iso_rasterFromNormal.tif'
            #lbl_path_iso = '/scratch2/jbaumer/2022_MasterThesis/datasets/dataset_testset/BB1/20200116/final_20200116_extract_snowclass_iso_rasterFromIso.tif'

            with rasterio.open(img_path) as src:
                img = src.read()
                img = img/255
            with rasterio.open(label_path) as src:
                image = rasterio.open(img_path)
                # in case lbl is not 6000x4000, fill up with zeros correctly 
                lbl = get_full_resolution_label(image, src)
                # lbl = src.read()

            lbl[lbl==3] = 2


            img = torch.as_tensor(img).to(self.device)
            lbl = torch.as_tensor(lbl).to(self.device)
            
            # make sure they are divisible by 32 (e.g. needed for U-Net)
            img = transforms.functional.crop(img, top=0, left=0, height=32*(img.shape[1]//32), width=32*(img.shape[2]//32))  # ev everywhere -30*32
            lbl = transforms.functional.crop(lbl, top=0, left=0, height=32*(lbl.shape[1]//32), width=32*(lbl.shape[2]//32))  # TODO: change back to 32 - get rid of memory issues

            img = self.normalize(img)

            return img.float(), lbl.long()


        else:  # train / val
            label_shape = rasterio.open(label_path).shape
            xrand = random.randint(0, label_shape[1] - self.patch_size[1])
            yrand = random.randint(0, label_shape[0] - self.patch_size[0])

            # get image and label patches
            lbl_patch, xshift, yshift = get_label_patch_and_tf(label_path, self.patch_size, x_rand=xrand, y_rand=yrand)

            img_patch = get_image_patch(img_path, xshift, yshift, self.patch_size, x_rand=xrand, y_rand=yrand)
            img_patch = img_patch/255

            i = torch.as_tensor(img_patch).to(self.device)  # if errors, change from torch.Tensor() to torch.as_tensor()
            l = torch.as_tensor(lbl_patch).to(self.device)

            # return i.float(), l.long()

            # cat img and lbl tensors together, apply train augmentation, the split again to img, lbl
            # note:
            #   ColorJitter, GaussianBlur and Normalization only for img
            #   RandomHorizontalFlip applied on img and label with given probability
            if self.mode == 'train':
                imglbl = torch.cat((i, l), dim=0)  # torch.Size([4, patch_size[0], patch_size[1]])
                i, l = data_augmentation(imglbl, self.train_augmentation, self.gblur, self.normalize, self.coljit)
            
            else:
                i = self.normalize(i)

            return i.float(), l.long()  # img.dtype: float, range 0 and 1 / lbl.dtype: long, either 0 or 1 or 2

        """
        # test plots:
        plt.imshow(np.transpose(img.cpu().numpy(), (1,2,0)).squeeze())
        plt.imshow(np.transpose(lbl.cpu().numpy(), (1,2,0)).squeeze())
        # plt.imsave()
        
        # plot images and labels
        plt.figure()
        f, axarr = plt.subplots(2, 1) #subplots(r,c) provide the no. of rows and columns
        axarr[0].set_title(f'Image Patch {self.patch_size}')
        axarr[0].imshow(np.transpose(i.cpu().numpy(), (1,2,0)))
        axarr[1].set_title(f'Label Patch {self.patch_size}')
        axarr[1].imshow(np.transpose(l.cpu().numpy(), (1,2,0)))
        plt.show()
        """


    def get_balancedness():
        pass


if __name__=='__main__':

    all = ['Buelenberg_1', 'Buelenberg_2', 'Giementaelli_1', 'Giementaelli_2', 'Giementaelli_3', 'Luksch_1', 'Luksch_2', 'Sattel_1', 'Sattel_2', 'Sattel_3', 'Stillberg_1', 'Stillberg_2', 'Stillberg_3']
    some = ['Sattel_1', 'Stillberg_1', 'Stillberg_2', 'Buelenberg_1']

    x1 = DischmaSet_segmentation(root="/net/pf-pc20/scratch2/jbaumer/2022_MasterThesis/datasets/dataset_complete/", stat_cam_lst=['Buelenberg_2'], mode='train')
    x2 = DischmaSet_segmentation(root="/net/pf-pc20/scratch2/jbaumer/2022_MasterThesis/datasets/dataset_complete/", stat_cam_lst=['Buelenberg_2'], mode='val')    
    x3 = DischmaSet_segmentation(root="/net/pf-pc20/scratch2/jbaumer/2022_MasterThesis/datasets/dataset_complete/", stat_cam_lst=['Buelenberg_2'], mode='test')
    x4 = DischmaSet_segmentation(root="/net/pf-pc20/scratch2/jbaumer/2022_MasterThesis/datasets/dataset_complete/", stat_cam_lst=['Buelenberg_2'], mode='baseline')

    #x1 = DischmaSet_segmentation(root='../datasets/dataset_complete/', stat_cam_lst=['Buelenberg_2'], mode='train')

    i, l = x1.__getitem__(0)
    i, l = x2.__getitem__(0)
    i, l = x3.__getitem__(0)
    i, l = x4.__getitem__(0)



    x = DischmaSet_segmentation(root='../datasets/dataset_complete/', stat_cam_lst=['Buelenberg_1'])
    #for n in range(30, 100):
    #    img, lbl = x.__getitem__(n)
    img, lbl = x.__getitem__(50)
    img, lbl = x.__getitem__(50)
    img, lbl = x.__getitem__(50)
    img, lbl = x.__getitem__(50)
    img, lbl = x.__getitem__(50)
    img, lbl = x.__getitem__(50)
    img, lbl = x.__getitem__(50)
    img, lbl = x.__getitem__(50)
    img, lbl = x.__getitem__(50)  # at this breakpoint: img/lbl_ehafner_v01.jpg/tif (if getitem was already called 8 times)
    img, lbl = x.__getitem__(50)  # at this breakpoint: img/lbl_ehafner_v02.jpg/tif

    #plt.imshow(np.transpose(img.cpu().numpy(), (1,2,0)).squeeze())
    #plt.imshow(np.transpose(lbl.cpu().numpy(), (1,2,0)).squeeze())

    print()

