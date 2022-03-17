#%% Unsupervised image clustering - CNN & k-means
# Bickel,V.T. - 2020 - bickel@vaw.baug.ethz.ch
# used for creating GT that will be used for training the CNN model

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from sklearn.cluster import KMeans
import os, shutil, glob, os.path

print("Imports successful")

#%% I

image.LOAD_TRUNCATED_IMAGES = True
model = VGG16(weights='imagenet', include_top=False)
#model = VGG16(include_top=True, weights="imagenet", classes=1000)

# Variables
imdir = './input/'
targetdir = "./output/"
number_clusters = 2 #TODO user specified / acc. to elisabeth, use 4 or 5 - then check manually which directories are foggy / which are not foggy

#%% II

# Loop over files and get features
filelist = glob.glob(os.path.join(imdir, '*.jpg')) #TODO adapt to data
filelist.sort()
featurelist = []
for i, imagepath in enumerate(filelist):
    print("    Status: %s / %s" %(i, len(filelist)), end="\r")
    img = image.load_img(imagepath, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = np.array(model.predict(img_data))
    featurelist.append(features.flatten())

#%% III

# Clustering
kmeans = KMeans(n_clusters = number_clusters, random_state=0).fit(np.array(featurelist))
try:
    os.makedirs(targetdir)
except OSError:
    pass

print(kmeans)

# Copy with cluster name
print("\n")
for i, m in enumerate(kmeans.labels_):
    print("    Copy: %s / %s" %(i, len(kmeans.labels_)), end="\r")
    shutil.copy(filelist[i], targetdir + str(m) + "_" + str(filelist[i])[8:])
