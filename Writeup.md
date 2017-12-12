## Writeup Template

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./pic/Resource.png
[image2]: ./pic/HOG_Feature.png
[image3]: ./pic/Sliding_Win1.png
[image4]: ./pic/Heat_Map1.png
[image5]: ./pic/Sliding_Win2.png
[image6]: ./pic/Heat_Map2.png
[video1]: ./op_project_v2.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Histogram of Oriented Gradients (HOG)

#### 1. Extract HOG features from the training images.

The code for all the feature extractions are in utils.py file. 

The resources have been labeled as car or non-car: 

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. HOG parameters.

Different combinations of parameters has been tried, the final version is as following:

At the very begining, I lean to use all the channels for hog feature extraction, but it will take too much time (3 times compared to single channel). And actually the most useful info is contained in the Y channel, single Y channel can achieve almost the same accuracy as 3 channel (0.97~0.98)
__________________________________________________________________
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
__________________________________________________________________
orient = 9  # HOG orientations
__________________________________________________________________
pix_per_cell = 8 # HOG pixels per cell
__________________________________________________________________
cell_per_block = 2 # HOG cells per block
__________________________________________________________________
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
__________________________________________________________________

#### 3. Classifier Training.

I trained a linear SVM using all spatial color feature, hist color feature and HOG feature, the parameters are as following:
__________________________________________________________________
spatial_size = (16, 16) # Spatial binning dimensions
__________________________________________________________________
hist_bins = 16    # Number of histogram bins
__________________________________________________________________
Final accuracy is around 0.97~0.98

### Sliding Window Search

The function definition of sliding window search is in utils.py, this function has been used in both test_plot.py and pipeline.py.

The searching window is limited to following size:
__________________________________________________________________
x_start_stop = [200, 1280] # Min and max in y to search in slide_window()
__________________________________________________________________
y_start_stop = [400, 660] 
__________________________________________________________________

Ultimately I searched on 3 scales using YCrCb Y-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
---

### Video Implementation

Here's a [link to my video result](./op_project_v2.mp4)

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

This method works, but I still have many false positives, in order to improve the detection result. I recorded hot window list from last 10 frames, then generate a heat_map based on these data. The result is much more stable now. 

---

### Discussion

Right I still have 2 problems:
1. False positive still happens (only once)
2. The size of the bounding box keep changing

Potential solution:
1. Set size threshold of the detected bounding box, normally the false positive bounding box is relatively small
2. Store bounding box of last 5 frames, calculate an average bounding box.

