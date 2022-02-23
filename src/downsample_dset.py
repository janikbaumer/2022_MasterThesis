import os
import shutil
from PIL import Image
import torchvision
from torchvision.transforms import functional as F
from torchvision.utils import save_image
import torch 

def ds_img(path_old, path_old_ds, img_name, factor=10):
    image = torchvision.io.read_image(path=path_old)
    image = image.float()  # convert uint8 to float32

    # write unsampled image to disk (to new destination)
    #save_image(tensor=image, fp='/scratch2/unsampled.jpg')  # save as jpg
    #torch.save(obj=image, f=f'/scratch2/unsampled.pkl')  # save as pkl
    #save_image(tensor=image, fp='/scratch2/unsampled.png')  # save as jpg


    shp = image.shape  # torch.Size([3, 6000, 4000])
    shp = tuple(image.shape[-2:])  # (6000, 4000)
    shp_new = tuple(int(x/factor) for x in shp)
    image_ds = F.resize(img=image, size=shp_new)

    # save new image
    #save_image(tensor=image_ds, fp='/scratch2/sampled.jpg')
    #torch.save(obj=image_ds, f='/scratch2/sampled.pkl')
    save_image(tensor=image_ds, fp=path_old_ds)


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
                ds_img(path_old=path_full_old_img, path_old_ds=path_full_old_img_ds, img_name=file_without_jpg_extenstion, factor=10)

                # move downsampled image to dir dataset_downsampled
                shutil.move(src=path_full_old_img_ds, dst=path_full_new_img_ds)

        print(f'{CamX_station} images DONE.')

# downsample_composites()
