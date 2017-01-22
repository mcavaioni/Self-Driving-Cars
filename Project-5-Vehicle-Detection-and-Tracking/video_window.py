
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import imutils
from imutils.object_detection import non_max_suppression
import glob
import time
import scipy
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
saved_clf = joblib.load('saved_model3.pkl') 
saved_scaler = joblib.load('saved_scaler3.pkl')

#Sliding window:
# for count in range(720,730):
# image = mpimg.imread('/Users/michelecavaioni/Flatiron/My-Projects/Udacity (Self Driving Car)/Project #5 (Vehicle Detection and Tracking/my_img/frame%d.jpg' %count)
# image = mpimg.imread('/Users/michelecavaioni/Flatiron/My-Projects/Udacity (Self Driving Car)/Project #5 (Vehicle Detection and Tracking/my_img/frame870.jpg')


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

    
    
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
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


def single_detected_boxes(image, windows):
  detected_boxes=[]
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
      # one_wind_flatten = sum(one_wind, ())
      # detected_boxes.append(one_wind_flatten)
      detected_boxes.append(one_wind)
  print(detected_boxes)
  return detected_boxes


def color_blue_filter(image):
  B = image[:,:,2]
  thresh = (254, 255)
  binary = np.zeros_like(B)
  binary[(B > thresh[0]) & (B <= thresh[1])] = 1
  return binary

def contours_list(image):
  windows_medium = slide_window(image, x_start_stop=[200, 1280], y_start_stop=[400, 500], 
                        xy_window=(120, 80), xy_overlap=(0.8, 0.8))

  windows_small = slide_window(image, x_start_stop=[700, 1100], y_start_stop=[400, 480], 
                      xy_window=(50, 40), xy_overlap=(0.65, 0.65))

  windows_big = slide_window(image, x_start_stop=[200, 1280], y_start_stop=[400, 650], 
                      xy_window=(200, 200), xy_overlap=(0.8, 0.8))

  combined = windows_medium+windows_big+windows_small

  detected_boxes = single_detected_boxes(image, combined)

  # if detected_boxes != []:
  image_copy = np.zeros_like(image)
  #fill up the intersecting rectangles so to create a full area:
  window_filled = draw_boxes(image_copy, detected_boxes, color=(0, 0, 255), thick=-1)
  #filter for blue color (color of the filled area) and create binary image:
  binary = color_blue_filter(window_filled)

  ret, thresh = cv2.threshold(binary.astype(np.uint8) * 255, 127, 255, 0)
  im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  return contours
  # else: 
  #   return 0
import collections
class Detection():
  def __init__(self, centroid, frame, x, y, h, w):
      self.detected = False 
      self.centroid = centroid
      self.frame = 1
      self.count = 0
      self.x = x
      self.y =y
      self.h = h
      self.w = w

# # single image: (comment line 183, 184; uncomment line 185)
frame_count=0
frame_detections =[]
centre_count = collections.deque(maxlen=5)
width_count = collections.deque(maxlen=5)
height_count = collections.deque(maxlen=5)
x_count = collections.deque(maxlen=5)
y_count = collections.deque(maxlen=5)

def pipeline(image):
  global frame_count, centroid, contours, count, frame_detections, past_frame_detections, centre_count, width_count, height_count, x_count, y_count
  frame_count+=1


  if frame_count == 1:
    contours = contours_list(image)
    if contours != 0:
    #draw rectangle over the detected contour areas and centroid of rectangle:
      for contour in contours:
        cnt = contour
        x,y,w,h = cv2.boundingRect(cnt)
        centroidx = (x+w/2)
        centroidy = (y+h/2)
        centroid = (centroidx,centroidy)
        cv2.rectangle(image,(x,y),(x+w,y+h),(192,192,192),2)

        current_detection = Detection(centroid, frame_count, x, y, h, w)
        frame_detections.append(current_detection)
        # cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        # cv2.circle(image, (centroidx, centroidy), 10, (0, 255, 0), -1)
        # print(centroid)
        # print(frame_detections[0].centroid)
      past_frame_detections = frame_detections
    else:
      past_frame_detections = []

  else:
    contours = contours_list(image)
    if contours != []:
      frame_detections = []
      for contour in contours:
        cnt = contour
        x,y,w,h = cv2.boundingRect(cnt)
        centroidx = (x+w/2)
        centroidy = (y+h/2)
        centroid = (centroidx,centroidy)
        cv2.rectangle(image,(x,y),(x+w,y+h),(192,192,192),2)

        current_detection = Detection(centroid, frame_count, x, y, h, w)
        frame_detections.append(current_detection)
      print(len(contours))  
      print(len(frame_detections))
      if past_frame_detections != []:
        for past_frame_det in past_frame_detections:
          for curr_frame_det in frame_detections:
            print(curr_frame_det)
            if len(centre_count)<5 and (frame_count%5 != 0) and abs(past_frame_det.centroid[0]-curr_frame_det.centroid[0])<200 and abs(past_frame_det.centroid[1]-curr_frame_det.centroid[1])<200:
              print('putting stuff')
              print(len(centre_count))
              centre_count.append(curr_frame_det.centroid[0])
              width_count.append(curr_frame_det.w)
              height_count.append(curr_frame_det.h)
              x_count.append(curr_frame_det.x)
              y_count.append(curr_frame_det.y)

            elif (frame_count%5 == 0):
              if len(centre_count)==5:  
                print('here iam mean')
                centre_mean = int(np.mean(centre_count))
                widht_mean = int(np.mean(width_count))
                height_mean = int(np.mean(height_count))
                x_mean = int(np.mean(x_count))
                y_mean = int(np.mean(y_count))

                cv2.circle(image, (x_mean, y_mean), 10, (0, 255, 0), -1)
                cv2.rectangle(image,(x_mean,y_mean),(x_mean+widht_mean,y_mean+height_mean),(0,255,0),2)

                centre_count.clear()
                width_count.clear()
                height_count.clear()
                x_count.clear()
                y_count.clear()

              elif (len(centre_count)<5):
                print('clearing all')
                centre_count.clear()
                width_count.clear()
                height_count.clear()
                x_count.clear()
                y_count.clear()

            else:
              pass

        past_frame_detections = frame_detections
      else:
        past_frame_detections = frame_detections
    else:
      past_frame_detections = []
  return image
####################

#VIDEO:
from moviepy.editor import VideoFileClip
from IPython.display import HTML
vid_output = 'detection_video.mp4'
clip1 = VideoFileClip('/Users/michelecavaioni/Flatiron/My-Projects/Udacity (Self Driving Car)/Project #5 (Vehicle Detection and Tracking/project_video.mp4')
vid_clip = clip1.fl_image(pipeline) 
vid_clip.write_videofile(vid_output, audio=False)
