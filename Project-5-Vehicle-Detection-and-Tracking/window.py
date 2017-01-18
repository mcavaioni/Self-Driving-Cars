
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import imutils
from imutils.object_detection import non_max_suppression
import glob
import time
from sklearn.svm import LinearSVC, SVC
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split

def color_hist(img, nbins=32, bins_range=(0, 256)):
  '''
  computes histogram of color features 
  '''
  # Compute the histogram of the color channels separately
  channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
  channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
  channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
  # Concatenate the histograms into a single feature vector
  hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
  # Return the individual histograms, bin_centers and feature vector
  return hist_features


def bin_spatial(img, size=(32, 32)):
  '''
  computes binned color features 
  '''
  # Use cv2.resize().ravel() to create the feature vector
  features = cv2.resize(img, size).ravel() 
  # Return the feature vector
  return features

#NOTE: input image is grayscale
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    '''
    returns HOG features and visualization. Input is a grascale image.
    '''
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# # Plot HOG:
# gray = cv2.cvtColor((cars[0]), cv2.COLOR_RGB2GRAY)
# features, hog_image = get_hog_features(gray, orient=9, pix_per_cell=8, cell_per_block=2,vis=True,feature_vec=False)
# plt.imshow(hog_image, cmap='gray')
# plt.show()

def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256), orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    '''
    Combines features from a list of images. 
    Features include: binned colors, histogram of colors, histogram of oriented gradient (HOG)
    '''
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image=file
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        else: feature_image = np.copy(image)      
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        # Call get_hog_features() with vis=False, feature_vec=True
        hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features, hog_features)))
    # Return list of feature vectors
    return features

from sklearn.externals import joblib
saved_clf = joblib.load('saved_model.pkl') 
saved_scaler = joblib.load('saved_scaler.pkl')

#Sliding window:

image = mpimg.imread('/Users/michelecavaioni/Flatiron/My-Projects/Udacity (Self Driving Car)/Project #5 (Vehicle Detection and Tracking/test5.jpg')


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    '''
    draws rectangular boxes based on given vertices
    '''
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    '''
    creates sliding windows throughout the whole image. It outputs a list of vertices to be used to create rectangular boxes
    '''
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop==[None, None]:
      x_start_stop = [0,img.shape[1]]
    if y_start_stop==[None, None]:
      y_start_stop = [0,img.shape[0]]
    # Compute the span of the region to be searched   
    
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    num_wind_x = np.int((x_start_stop[1] - x_start_stop[0])/nx_pix_per_step)-1
    num_wind_y = np.int((y_start_stop[1] - y_start_stop[0])/ny_pix_per_step)-1
    # num_wind_x = int(x_start_stop[1]/(xy_window[0]*xy_overlap[0]))-1
    # num_wind_y = int(y_start_stop[1]/(xy_window[1]*xy_overlap[1]))-1
    
    
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    #     Note: you could vectorize this step, but in practice
    #     you'll be considering windows one by one with your
    #     classifier, so looping makes sense
        # Calculate each window position
        # Append window position to list
    for iter_y in range(num_wind_y):
      for iter_x in range(num_wind_x):
            beg_x = nx_pix_per_step*iter_x + x_start_stop[0]
            beg_y = ny_pix_per_step*iter_y + y_start_stop[0]
            end_x = beg_x + xy_window[0]
            end_y = beg_y + xy_window[1]
            points = ((beg_x,beg_y),(end_x,end_y))
            window_list.append(points)
    # Return the list of windows
    return window_list

windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 500], 
                    xy_window=(120, 80), xy_overlap=(0.8, 0.8))



def check_single_wind(image):
  detected_boxes = []
  for one_wind in windows:
    # one_wind = [one_wind]
    y1 = one_wind[0][1]
    y2 = one_wind[1][1]
    x1 = one_wind[0][0]
    x2 = one_wind[1][0]
    crop_img = image[y1:y2,x1:x2]
    resized = cv2.resize(crop_img,(64,64))
    res_feat = extract_features([resized], cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256), orient=9, 
                        pix_per_cell=4, cell_per_block=2, hog_channel=0)
    res_feat_norm = saved_scaler.transform(res_feat)
    new_prediction = saved_clf.predict(res_feat_norm)
    confidence_scores = saved_clf.decision_function(res_feat_norm)

    if (new_prediction[0]) ==1.0 and confidence_scores>[1.0]:
      # detected_boxes.append(one_wind)
      # one_wind_flatten = [element for tupl in one_wind for element in tupl]
      # print(confidence_scores)
  # window_img = draw_boxes(image, detected_boxes, color=(0, 0, 255), thick=6)
  # plt.imshow(window_img)
  # plt.show()
  # return window_img
  ###########
  #working suppression but not great
      one_wind_flatten = sum(one_wind, ())
      detected_boxes.append(one_wind_flatten)
  bounding_boxes = []
  detected_boxes = np.array(detected_boxes)
  pick = non_max_suppression(detected_boxes, probs=None, overlapThresh=0)
  for (xA, yA, xB, yB) in pick:
    xmin = min(xA,xB)
    ymin = min(yA,yB)
    xmax = max(xA,xB)
    ymax = max(yA,yB)
    bounding_boxes.append(((xmin,ymin),(xmax,ymax)))
  window_img = draw_boxes(image, bounding_boxes, color=(0, 0, 255), thick=6)
  # plt.imshow(window_img)
  # plt.show()
  return window_img
  #######
check_single_wind(image)

#VIDEO:
from moviepy.editor import VideoFileClip
from IPython.display import HTML
vid_output = 'detection_video.mp4'
clip1 = VideoFileClip('/Users/michelecavaioni/Flatiron/My-Projects/Udacity (Self Driving Car)/Project #5 (Vehicle Detection and Tracking/project_video.mp4')
vid_clip = clip1.fl_image(check_single_wind) 
vid_clip.write_videofile(vid_output, audio=False)

# plt.show()

###########
# one_wind = [(windows[1])]
#crop image
# crop_img = img[y: y + h, x: x + w]
# y1 = one_wind[0][0][1]
# y2 = one_wind[0][1][1]
# x1 = one_wind[0][0][0]
# x2 = one_wind[0][1][0]
# crop_img = image[y1:y2,x1:x2]
#  RESIZE TO 64x64 BEFORE FEEDING CLASSIFIER

# #draw the boxes:                       
# window_img = draw_boxes(image, one_wind, color=(0, 0, 255), thick=6)                    
# plt.imshow(window_img)
# # plt.imshow(crop_img)
# plt.show()