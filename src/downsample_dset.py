import os
import shutil
from PIL import Image
import torchvision
from torchvision.transforms import functional as F
from torchvision.utils import save_image
import torch
from torchvision.transforms.transforms import ToTensor, ToPILImage
import torch.nn.functional as F

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def ds_img(path_old, path_old_ds, factor=10):
    print(path_old)
    cvt_pil_to_tens = ToTensor()
    cvt_tens_to_pil = ToPILImage()
    try:
        img_pil = Image.open(fp=path_old)
    except:
        return None
    img_tensor = cvt_pil_to_tens(img_pil)  # shape 3,4000,6000, values in range [0,1]
    img_tensor = img_tensor[None, ...]  # add first dim (needed for F.interpolate), new shape 1,3,4000,6000
    img_tensor_ds = F.interpolate(input=img_tensor, scale_factor=(1/factor, 1/factor))  # shape 1, 3, 4000/factor, 6000/factor
    img_tensor_ds_squeezed = torch.squeeze(input=img_tensor_ds, dim=0)  # remove first dim if it is one, new shape 3, 4000/factor, 6000/factor
    img_ds = cvt_tens_to_pil(img_tensor_ds_squeezed)
    img_ds.save(fp=path_old_ds)  # format is inferred from extension (png)





def downsample_composites(ds_factor=10):
    """
    function creates a subsampled dataset from the directory dataset_complete
    this fct considering the subdir 'Composites'
    directory 'dataset_downsampled' with all subdirs has to be created manually
    to get same dir sub-structure, follow: https://stackoverflow.com/questions/4073969/copy-folder-structure-without-files-from-one-location-to-another
    !!! only run code once (to one specific directory), no checking for duplicates (for efficiency reason)!
    """
    
    PATH_COMPOSITES_COMPLETE = '../datasets/dataset_complete/Composites'
    PATH_COMPOSITES_DOWNSAMPLED = '../datasets/dataset_downsampled/Composites'
    
    print('Start downscaling composite images...')
    for CamX_station in os.listdir(f'{PATH_COMPOSITES_COMPLETE}'):
        print(f'Start downscaling {CamX_station} images...')  # CamX_station = e.g. 'Cam1_Buelenberg'

        path_cam_x_station_old = f'{PATH_COMPOSITES_COMPLETE}/{CamX_station}'
        path_cam_x_station_new = f'{PATH_COMPOSITES_DOWNSAMPLED}/{CamX_station}'

        for file in os.listdir(path_cam_x_station_old):  # file: mostly txt and jpg files

            path_full_old_noimg = f'{path_cam_x_station_old}/{file}'
            path_full_new_noimg = f'{path_cam_x_station_new}/{file}'

            # copy all non .jpg files directly (.txt files etc) (w\o downsampling)
            if not (file.endswith('.jpg') or file.endswith('.png')):
                shutil.copyfile(src=path_full_old_noimg, dst=path_full_new_noimg)

            if file.endswith('.jpg'):
                file_without_jpg_extenstion = file[:-4]
                file_with_png_extension = f'{file_without_jpg_extenstion}.png'

                path_full_old_img = f'{path_cam_x_station_old}/{file}'
                path_full_old_img_ds = f'{path_cam_x_station_old}/{file_with_png_extension}'
                path_full_new_img_ds = f'{path_cam_x_station_new}/{file_with_png_extension}'

                # downsample image
                ds_img(path_old=path_full_old_img, path_old_ds=path_full_old_img_ds, factor=10)

                # move downsampled image to dir dataset_downsampled
                shutil.move(src=path_full_old_img_ds, dst=path_full_new_img_ds)

        print(f'{CamX_station} images DONE.')




def downsample_DischmaCams(ds_factor=10):
    """
    function creates a subsampled dataset from the directory dataset_complete
    this fct considering the subdir 'DischmaCams'
    directory 'dataset_downsampled' with all subdirs has to be created manually
    to get same dir sub-structure, follow: https://stackoverflow.com/questions/4073969/copy-folder-structure-without-files-from-one-location-to-another
    !!! only run code once (to one specific directory), no checking for duplicates (for efficiency reason)!
    """
    
    PATH_DISCHMACAMS_COMPLETE = '../datasets/dataset_complete/DischmaCams'
    PATH_DISCHMACAMS_DOWNSAMPLED = '../datasets/dataset_downsampled/DischmaCams'

    print('Start downscaling DischmaCams images...')

    for station in os.listdir(PATH_DISCHMACAMS_COMPLETE):
        path_station_old = f'{PATH_DISCHMACAMS_COMPLETE}/{station}'        
        path_station_new = f'{PATH_DISCHMACAMS_DOWNSAMPLED}/{station}'

        for camera in os.listdir(path_station_old):
            print(f'Start downscaling {station}_{camera} images...')  # {station}_{camera} = e.g. 'Buelenberg_1'

            path_station_camera_old = f'{path_station_old}/{camera}'
            path_station_camera_new = f'{path_station_new}/{camera}'
            

            for file in os.listdir(path_station_camera_old):  # file: mostly txt and jpg files
                
                path_full_old_noimg = f'{path_station_camera_old}/{file}'
                path_full_new_noimg = f'{path_station_camera_new}/{file}'

                # copy all non .jpg files directly (.txt files etc) (w\o downsampling)
                if not (file.endswith('.jpg') or file.endswith('.png')):
                    shutil.copyfile(src=path_full_old_noimg, dst=path_full_new_noimg)
                
                if file.endswith('.jpg'):
                    file_without_jpg_extenstion = file[:-4]
                    file_with_png_extension = f'{file_without_jpg_extenstion}.png'

                    path_full_old_img = f'{path_station_camera_old}/{file}'
                    path_full_old_img_ds = f'{path_station_camera_old}/{file_with_png_extension}'
                    path_full_new_img_ds = f'{path_station_camera_new}/{file_with_png_extension}'


                # downsample image
                ds_img(path_old=path_full_old_img, path_old_ds=path_full_old_img_ds, factor=10)

                # move downsampled image to dir dataset_downsampled
                shutil.move(src=path_full_old_img_ds, dst=path_full_new_img_ds)

        print(f'{station} images DONE.')



def downsample_final_workdirs():
    """
    function creates a subsampled dataset from the directory dataset_complete
    this fct considering the subdirs 'final_workdir_STATION_CamX'
    directory 'dataset_downsampled' with all subdirs has to be created manually
    to get same dir sub-structure, follow: https://stackoverflow.com/questions/4073969/copy-folder-structure-without-files-from-one-location-to-another
    !!! can be run multiple times, duplicates will be overridden (for efficiency reason)!
    """


    # TODO: make sure there are no multi channel tif images (else convert them first to single channel, or remove (if only few))

    # PATH_COMPLETE = '../datasets/dataset_devel'
    PATH_COMPLETE = '../datasets/dataset_complete'

    # PATH_DOWNSAMPLED = '../datasets/dataset_downsampled_devel'
    PATH_DOWNSAMPLED = '../datasets/dataset_downsampled'

    n_fail = 0

    for final_workdir_station_CamX in os.listdir(f'{PATH_COMPLETE}'):  # TODO: snowmap all
        if final_workdir_station_CamX.startswith('final_workdir_Fog_threshold'):  # to not consider Composites and DischmaCams folder
            print(f'start downsampling with images from {final_workdir_station_CamX}')

            path_old = os.path.join(PATH_COMPLETE, final_workdir_station_CamX)
            path_new = os.path.join(PATH_DOWNSAMPLED, final_workdir_station_CamX)

            for file in os.listdir(path_old):
                file_old = os.path.join(path_old, file)
                file_old_wo_ext = file_old[:-4]
                file_old_ds = file_old_wo_ext + '_downsampled.tif'

                file_new = os.path.join(path_new, file)

                # copy all non .tif files (.txt files etc) directly (w\o downsampling)
                if not (file.endswith('.tif')):  # or file.endswith('.tiff')):
                    shutil.copyfile(src=file_old, dst=file_new)

                if (file.endswith('.tif')): # or file.endswith('.tiff')): - not needed (there are no tiff files, only tif)
                    # downsample image
                    ds_img(path_old=file_old, path_old_ds=file_old_ds, factor=10)

                    # move downsampled image to dir dataset_downsampled
                    try:
                        shutil.move(src=file_old_ds, dst=file_new)
                    except:
                        print(f'file {file} could not be downsampled')
                        n_fail += 1
                        continue

            print(f'DONE with images from {final_workdir_station_CamX}')

    print('number of images not downsampled: ', n_fail)

# downsample_DischmaCams()
# downsample_composites()
downsample_final_workdirs()
