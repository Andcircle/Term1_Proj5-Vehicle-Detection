import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
import cv2
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import glob
from utils import *
from sklearn.cross_validation import train_test_split
import sklearn.utils as sk

color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

def trainSVC(color_space=color_space, 
             spatial_size=spatial_size, hist_bins=hist_bins, 
             orient=orient, pix_per_cell=pix_per_cell, 
             cell_per_block=cell_per_block, 
             hog_channel=hog_channel, spatial_feat=spatial_feat, 
             hist_feat=hist_feat, hog_feat=hog_feat):
    cars = []
    notcars = []
    
    images = glob.glob('./non-vehicles/Extras/*.png')
    print(len(images))
    for image in images:
        notcars.append(image)
        
    images = glob.glob('./non-vehicles/GTI/*.png')
    print(len(images))
    for image in images:
        notcars.append(image)
        
    print('Number of non-vehicle pics:', len(notcars))
        
    drs = glob.glob('./vehicles/*')
    for dr in drs:
        images = glob.glob(dr + '/*.png')
        for image in images:
            cars.append(image)
        
    print('Number of vehicle pics:', len(cars)) 
    #----------------------------------------- image = mpimg.imread(notcars[0]);
    #-------------------------------------------------------------- print(image)
    #------------------------------------------- image1 = mpimg.imread(cars[0]);
    #------------------------------------------------------------- print(image1)
    
    car_features = extract_features(cars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    
    
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    
    sf_X,sf_Y = sk.shuffle(scaled_X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        sf_X, sf_Y, test_size=0.2, random_state=rand_state)
    
    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))


    return svc,X_scaler

#----------------------------------------------------- svc,X_scaler = trainSVC()
