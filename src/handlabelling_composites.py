# jbaumer@ethz.ch
# 2022-03-24

# manual classification of images of the Dischma valley (Davos)
# if done with classification, make sure to end with button 'q' (do not interrupt program with Ctrl+C or similar !)
# foggy sky -> not foggy
# as soon as some snow is covered -> foggy
# too much reflection points on image, when img gets unclear -> foggy

# label images of 2020 / 2021
# if image is (F)oggy, press button 'f'
# if image is (N)ot foggy, press button 'n'
# to quit, press button 'q' (do not only exit with x-symbol on top right)

import os
import random
import tkinter
import cv2
from tkinter import messagebox

random.seed(42)

"""
# no more needed (image will not even be shown, by checking 'data' variable)
def write_to_file(string, textfile):
    # appends the line only to the textfile if it has not yet been classified before
    os.system(f"grep -q {string} {textfile} || echo {string} >> {textfile}")
"""

# get n labeled examples from test set
# cat manual_labels_isfoggy.txt | grep -e 2020[0-9][0-9]03 -e 2020[0-9][0-9]10 -e 2020[0-9][0-9]17 -e 2020[0-9][0-9]25 | wc -l

# get n images to be labeled (in total)
# ls 2020??03??????.* 2020??10??????.* 2020??17??????.* 2020??25??????.* | wc -l

STATCAM_LST = [
    'Buelenberg/1',
    'Buelenberg/2',
    'Giementaelli/1',
    'Giementaelli/2',
    'Giementaelli/3',
    'Luksch/1',
    'Luksch/2',
    'Sattel/1',
    'Sattel/2',
    'Sattel/3',
    'Stillberg/1',
    'Stillberg/2',
    'Stillberg/3'
    ]

# todo. label some more for test set - only 2 days a months are too little, as not all can be used
# because they also need to be found in txt files from SLF classification for comparison to baseline
# -> label days 03, 10, 17, 25 from each month in 2020

BASE_PATH = '../datasets/dataset_complete/Composites/'
CAM = 'Stillberg'  # may be changed
STAT = '3'  # may be changed
CAMSTAT_DIR = 'Cam' + STAT + '_' + CAM 

ext = ('.png', '.jpg')
manual_labels_file = os.path.join(BASE_PATH, CAMSTAT_DIR, 'manual_labels_isfoggy.txt')
filelist = os.listdir(os.path.join(BASE_PATH, CAMSTAT_DIR))
filelist = [ele for ele in filelist if ele.endswith('.jpg')]
random.shuffle(filelist)

# read whole txt file (to check later whether img has already been labelled)
if os.path.isfile(manual_labels_file):
    with open(file=manual_labels_file, mode='r') as txtfile:
        data = txtfile.read()
else:
    data = ''

with open(file=manual_labels_file, mode='a') as txtfile:
    for file in filelist:
        quit, cont = False, False

        yr = file[6:10]
        day = file[12:14]

        # check if this image has already been classified manually (if line exists in txt file)
        # if following two lines are uncommented, then images may be classified multiple times - manual_labels_isfoggy.txt files would have to be checked for duplicates
        # with these lines, no duplicates should occur (except ev if quitted incorrectly - not with 'q')
        if file[0:14] in data:  # this image was already labelled, go to next one
            continue

        # if (file[0:4] == '2020' or file[0:4] == '2021') and file.endswith(ext):  # only label files of 2020/2021 
        #if (file[0:4] == '2021' and file[4:6] in ['01', '04', '07', '10']):  # only label data that will be used for training/validation (some months of 2021)
        if (file.endswith(ext)) and ((yr == '2020' and day in ['03', '10', '17', '25']) or (yr == '2021')):  # label test days from 2020 and all from 2021
            line = file + ' '
            usage_string = 'Usage: \n f: (F)oggy \n n: (N)ot foggy \n q: (Q)uit'
            img = cv2.imread(os.path.join(BASE_PATH, CAMSTAT_DIR, file))  # read a colour image from the working directory
            cv2.namedWindow(f'{file} - {usage_string} - manually classifying {CAMSTAT_DIR} images', cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f'{file} - {usage_string} - manually classifying {CAMSTAT_DIR} images', 600, 400)
            cv2.imshow(f'{file} - {usage_string} - manually classifying {CAMSTAT_DIR} images', img)  # display the original image

            key = cv2.waitKey(0) & 0xFF  # with param 0 (waitKey), the image is open till any key is pressed  /  0xFF only get last 8bits of the keys ( ord('...') ) to accomodate in case NumLock is activated (or similar)

            # KEYBOARD INTERACTIONS
            if key == ord('f'):
                line = line + '1\n'
                txtfile.write(line)
                cv2.destroyAllWindows()

            elif key == ord('n'):  
                line = line + '0\n'
                txtfile.write(line)
                cv2.destroyAllWindows()

            elif key == ord('q'):
                cv2.destroyAllWindows()
                quit = True

            else:  # error message for all other keys
                root = tkinter.Tk()
                messagebox.showinfo(title='Wrong Key Used', message=usage_string)
                root.destroy()
                cv2.destroyAllWindows()
                cont = True

        if quit == True:
            break
        if cont == True:
            continue

# print('everything from years 2021 (months: 01,04,07,10) classified (if not exited with q)')
print(f'everything from {CAMSTAT_DIR} and year 2020 (days: 03, 10, 17, 25) classified (if not exited with q)')
