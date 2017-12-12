import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
import cv2
import time
from sklearn.svm import LinearSVC
from utils import *
from train import trainSVC
import pickle


color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

#----------------------------- svc, X_scaler = trainSVC(color_space=color_space,
                         #------ spatial_size=spatial_size, hist_bins=hist_bins,
                         #------------ orient=orient, pix_per_cell=pix_per_cell,
                         #----------------------- cell_per_block=cell_per_block,
                         #-- hog_channel=hog_channel, spatial_feat=spatial_feat,
                         #-------------- hist_feat=hist_feat, hog_feat=hog_feat)
#------------------------------------------------------------------------------ 
#------------------------------- training_result = {"SVC":svc,"Scaler":X_scaler}
#---------- pickle.dump( training_result, open( "training_result_v1.p", "wb" ) )


#Add bounding boxes in this format, these are just example coordinates.
imaget = cv2.imread('./test_images/test1.jpg')
image = cv2.cvtColor(imaget, cv2.COLOR_BGR2RGB)
draw_image = np.copy(image)

# windows = slide_window(draw_image, x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                    #---------------- xy_window=(96, 96), xy_overlap=(0.5, 0.5))
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
# hot_windows = search_windows(draw_image, windows, svc, X_scaler, color_space=color_space,
                        #------- spatial_size=spatial_size, hist_bins=hist_bins,
                        #------------- orient=orient, pix_per_cell=pix_per_cell,
                        #------------------------ cell_per_block=cell_per_block,
                        #--- hog_channel=hog_channel, spatial_feat=spatial_feat,
                        #--------------- hist_feat=hist_feat, hog_feat=hog_feat)
#------------------------------------------------------------------------------ 
#-- window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)


training_result = pickle.load(open( "training_result.p", "rb" ))
svc = training_result["SVC"]
X_scaler = training_result["Scaler"]

x_start_stop = [200, image.shape[1]] # Min and max in y to search in slide_window()
y_start_stop = [400, 660] # Min and max in y to search in slide_window()

scales = [1,1.5,2]
iter_image = np.copy(image)

scaled_hot_windows=[]
for scale in scales:
    window_img,hot_windows = find_cars(draw_image, color_space,
                           x_start_stop, y_start_stop,
                           scale, svc, X_scaler, orient,
                           hog_channel,
                           pix_per_cell, cell_per_block,
                           spatial_size, hist_bins)

    iter_image = draw_boxes(iter_image, hot_windows, color=(scale*100, 0, 255-scale*100), thick=6)
    scaled_hot_windows.extend(hot_windows)

plt.imshow(iter_image)
pl.show()

threshhold=3
heatmap,labels = heat_map_labels(draw_image, scaled_hot_windows, threshhold)
label_img = draw_labeled_bboxes(draw_image, labels)

fig = plt.figure()
plt.subplot(121)
plt.imshow(label_img)
plt.title('Car Positions')
plt.subplot(122)
plt.imshow(heatmap, cmap='hot')
plt.title('Heat Map')
fig.tight_layout()
pl.show()
print(label_img.shape)








