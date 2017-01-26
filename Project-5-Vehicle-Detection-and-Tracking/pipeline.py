
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import scipy.misc
import imutils
import glob
import time
import scipy
from sklearn.svm import LinearSVC, SVC
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split

#Uploading the trained, saved model and scaler
from sklearn.externals import joblib
saved_clf = joblib.load('saved_model4.pkl') 
saved_scaler = joblib.load('saved_scaler4.pkl')


image = scipy.misc.imread('/Users/michelecavaioni/Flatiron/My-Projects/Udacity (Self Driving Car)/Project #5 (Vehicle Detection and Tracking/my_img/frame1000.jpg')


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


def extract_features(imgs, cspace='YCrCb', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256), orient=9, 
                        pix_per_cell=8, cell_per_block=2):
    '''
    Combines features from a list of images, after transforming the image in a different space (here transformed from RGB to YCrCb)
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
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
        else: feature_image = np.copy(image)      
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        # Call get_hog_features() with vis=False, feature_vec=True
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                orient, pix_per_cell, cell_per_block, 
                                vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features) 
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features, hog_features)))
    # Return list of feature vectors
    return features


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
  '''
  For each input windows it extracts the features (using "extract_features method") and runs a new_prediction
  using the saved classifier. Each window that is predicted as 1 (=car) and has a certain confidence confidence_scores
  is finally saved into a list of "detected boxes"
  '''
  # Initialize a list to append detected boxes positions to:
  detected_boxes=[]
  #looping through the sliding windows list:
  for one_wind in windows:
    y1 = one_wind[0][1]
    y2 = one_wind[1][1]
    x1 = one_wind[0][0]
    x2 = one_wind[1][0]
    #cropping the portion of the image included in each window:
    crop_img = image[y1:y2,x1:x2]
    resized = cv2.resize(crop_img,(64,64))
    #extracting the features inside each window:
    res_feat = extract_features([resized], cspace='YCrCb', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256), orient=9, 
                        pix_per_cell=8, cell_per_block=2)
    #use saved scaler to normalize the features:
    res_feat_norm = saved_scaler.transform(res_feat)
    #run prediction:
    new_prediction = saved_clf.predict(res_feat_norm)
    #define confidence score
    confidence_scores = saved_clf.decision_function(res_feat_norm)
    #append detected boxes positions if their prediction is 1.0 and confidence score is > 1.0
    if (new_prediction[0]) ==1.0 and confidence_scores>[1.0]:
      detected_boxes.append(one_wind)
  # Return the list of detected boxes
  return detected_boxes



def color_blue_filter(image):
  '''
  Applies blue filter to the image in order to detect just the detected boxes area (drawn in blue).
  It returns the binary result.
  '''
  B = image[:,:,2]
  thresh = (254, 255)
  binary = np.zeros_like(B)
  binary[(B > thresh[0]) & (B <= thresh[1])] = 1
  return binary


def contours_list(image):
  '''
  It returns a list of contours positions around detected boxes.
  '''
  #Map portion of the image with medium sliding windows
  windows_medium = slide_window(image, x_start_stop=[200, 1280], y_start_stop=[400, 500], 
                        xy_window=(120, 80), xy_overlap=(0.8, 0.8))
  #Map portion of the image with small sliding windows
  windows_small = slide_window(image, x_start_stop=[700, 1100], y_start_stop=[400, 480], 
                      xy_window=(50, 40), xy_overlap=(0.65, 0.65))
  #Map portion of the image with big sliding windows
  windows_big = slide_window(image, x_start_stop=[200, 1280], y_start_stop=[400, 650], 
                      xy_window=(200, 160), xy_overlap=(0.8, 0.8))

  #combine all the sliding windows positions
  combined = windows_medium+windows_big+windows_small

  #Run the prediction on each window and saved all the detected boxes
  detected_boxes = (single_detected_boxes(image, combined))

  image_copy = np.zeros_like(image)
  #fill up the area of the detected boxes rectangles so to create an area of blue color:
  window_filled = draw_boxes(image_copy, detected_boxes, color=(0, 0, 255), thick=-1)

  #It draws all the blue rectangles around the detected boxes (Uncomment below two lines to display)
  # for ((x1, y1), (x2, y2)) in detected_boxes:
  #   cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),2)

  #Filter only the blue areas
  binary = color_blue_filter(window_filled)

  #Threshold the binary image and detect contours around the filtered binary blue boxes:
  ret, thresh = cv2.threshold(binary.astype(np.uint8) * 255, 127, 255, 0)
  im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  return contours

#########################
#IMAGE PIPELINE:

def img_pipeline(image):
  t=time.time()
  
  contours = contours_list(image)
  #draw rectangle over the detected contour areas and centroid of rectangle:
  for contour in contours:
    cnt = contour
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    centroidx = (x+w/2)
    centroidy = (y+h/2)
    cv2.circle(image, (centroidx, centroidy), 10, (0, 255, 0), -1)
    centroid = (centroidx,centroidy)
    cv2.drawContours(image, contour, -1, (0,255,0), 3)

  plt.imshow(image)
  t2=time.time()
  print('Image running time',t2-t)
  plt.show()

# img_pipeline(image)

#########################
#VIDEO PIPELINE:
frame_count=0
frame_detections = []

def pipeline(image):
  '''
  It draws gray contours over detected boxes and then, to reduce false positives, it runs a heat map over 
  6 frames and identifies in green the boxes that are repeating in each frame.
  '''
  global frame_count, frame_detections

  frame_count+=1

  #every 7 frames it collects the prior 6 frame_detections and creates a heat map of the overlapping images. 
  if frame_count%7 == 0:
      if len(frame_detections) >= 6:
        heat_map = np.zeros_like(image)
        for (x1, y1, x2, y2) in frame_detections:
          heat_map[y1:y2,x1:x2] += 20

        #threshold the heat map to select only the intersecting areas that are appearing in each of the 6 prior frames
        _, binary = cv2.threshold(heat_map, 59, 255, cv2.THRESH_BINARY)
        #transform thresholded in binary and find contours around the intersecting areas
        binary_gray = cv2.cvtColor(binary, cv2.COLOR_RGB2GRAY)
        _, final_contours, _ = cv2.findContours(binary_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        #loop over the contours to draw rectangle and centroid
        for fin_cont in final_contours:
          x,y,w,h = cv2.boundingRect(fin_cont)
          centroidx = (x+w/2)
          centroidy = (y+h/2)
          cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),5)
          cv2.circle(image, (centroidx, centroidy), 10, (0, 255, 0), -1)
        #once it's done reset the frame_detection to empty in order to collect the following 6 frames
        frame_detections = []
      else:
        frame_detections = []
  else:
    #draw the rectangle around all the detected boxes
    contours = contours_list(image)
    for contour in contours:
      cnt = contour
      x,y,w,h = cv2.boundingRect(cnt)

      cv2.rectangle(image,(x,y),(x+w,y+h),(192,192,192),2)

      #append the contours of detected boxes for each frame 
      frame_detections.append((x,y,x+w,y+h))

  return image


from moviepy.editor import VideoFileClip
from IPython.display import HTML
vid_output = 'detection_video.mp4'
clip1 = VideoFileClip('/Users/michelecavaioni/Flatiron/My-Projects/Udacity (Self Driving Car)/Project #5 (Vehicle Detection and Tracking/project_video.mp4')
vid_clip = clip1.fl_image(pipeline) 
vid_clip.write_videofile(vid_output, audio=False)

