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
import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip

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

training_result = pickle.load(open( "training_result.p", "rb" ))
svc = training_result["SVC"]
X_scaler = training_result["Scaler"]

x_start_stop = [600, 1280] # Min and max in y to search in slide_window()
y_start_stop = [400, 660] # Min and max in y to search in slide_window()

threshhold=22
scales = [1,1.5,2]
frames = []
frame_num = 10

def process_image(img):
    draw_img = np.copy(img)
    iter_image = np.copy(img)
    
    scaled_hot_windows=[]
    for scale in scales:
        window_img,hot_windows = find_cars(draw_img, color_space,
                           x_start_stop, y_start_stop,
                           scale, svc, X_scaler, orient,
                           hog_channel,
                           pix_per_cell, cell_per_block,
                           spatial_size, hist_bins)
        scaled_hot_windows.extend(hot_windows)
        iter_image = draw_boxes(iter_image, hot_windows, color=(scale*100, 0, 255-scale*100), thick=4)
    
    if len(scaled_hot_windows)>0:
        frames.append(scaled_hot_windows)
        
    if len(frames)>frame_num:
        frames.pop(0)
    
    if len(frames)>0:  
        final_windows = np.concatenate(frames)
    else:
        final_windows = []
        
    
    heatmap,labels = heat_map_labels(draw_img, final_windows, threshhold)   
    label_img = draw_labeled_bboxes(draw_img, labels)
    
    return label_img

video_output1 = './output_videos/op_project_v2.mp4'
video_input1 = VideoFileClip('project_video.mp4')#.subclip(12,15)
processed_video = video_input1.fl_image(process_image)
processed_video.write_videofile(video_output1, audio=False)


