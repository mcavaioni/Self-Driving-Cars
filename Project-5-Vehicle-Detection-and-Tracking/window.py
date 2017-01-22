
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
image = mpimg.imread('/Users/michelecavaioni/Flatiron/My-Projects/Udacity (Self Driving Car)/Project #5 (Vehicle Detection and Tracking/my_img/frame750.jpg')


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

# ############
#currently not used: (only for one type of window dimension)
# windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 700], 
#                     xy_window=(200, 200), xy_overlap=(0.8, 0.8))

# def check_single_wind(image, windows):
#   detected_boxes = []
#   for one_wind in windows:
#     # one_wind = [one_wind]
#     y1 = one_wind[0][1]
#     y2 = one_wind[1][1]
#     x1 = one_wind[0][0]
#     x2 = one_wind[1][0]
#     crop_img = image[y1:y2,x1:x2]
#     resized = cv2.resize(crop_img,(64,64))
#     res_feat = extract_features([resized], cspace='RGB', spatial_size=(32, 32),
#                         hist_bins=32, hist_range=(0, 256), orient=9, 
#                         pix_per_cell=4, cell_per_block=2, hog_channel=0)
#     res_feat_norm = saved_scaler.transform(res_feat)
#     new_prediction = saved_clf.predict(res_feat_norm)
#     confidence_scores = saved_clf.decision_function(res_feat_norm)

#     if (new_prediction[0]) ==1.0 and confidence_scores>[1.0]:
#       detected_boxes.append(one_wind)
#       # print(confidence_scores)
#   window_img = draw_boxes(image, detected_boxes, color=(0, 0, 255), thick=6)
#   plt.imshow(window_img)
#   plt.show()
#   # return window_img
#   ###########
#   #working suppression but not great
#   #     one_wind_flatten = sum(one_wind, ())
#   #     detected_boxes.append(one_wind_flatten)
#   # bounding_boxes = []
#   # detected_boxes = np.array(detected_boxes)
#   # pick = non_max_suppression(detected_boxes, probs=None, overlapThresh=0)
#   # for (xA, yA, xB, yB) in pick:
#   #   xmin = min(xA,xB)
#   #   ymin = min(yA,yB)
#   #   xmax = max(xA,xB)
#   #   ymax = max(yA,yB)
#   #   bounding_boxes.append(((xmin,ymin),(xmax,ymax)))
#   # window_img = draw_boxes(image, bounding_boxes, color=(0, 0, 255), thick=6)
#   # plt.imshow(window_img)
#   # plt.show()
#   # return window_img
#   #######

# check_single_wind(image, windows)

# ############


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


def duplicate_suppression(image, detected_boxes):
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
  plt.imshow(window_img)
  plt.show()
  return window_img

def color_blue_filter(image):
  B = image[:,:,2]
  thresh = (254, 255)
  binary = np.zeros_like(B)
  # binary[(B == thresh[1])]=1
  binary[(B > thresh[0]) & (B <= thresh[1])] = 1
  return binary

# def pipeline(image):
#   # windows_medium = slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 500], 
#   #                     xy_window=(120, 80), xy_overlap=(0.8, 0.8))

#   # detected_boxes = (single_detected_boxes(image, windows_medium))

#   # windows_small = slide_window(image, x_start_stop=[500, 1000], y_start_stop=[400, 450], 
#   #                     xy_window=(20, 20), xy_overlap=(0.8, 0.8))
#   # detected_boxes = (single_detected_boxes(image, windows_small))

#   windows_semi = slide_window(image, x_start_stop=[500, 1000], y_start_stop=[400, 500], 
#                       xy_window=(50, 40), xy_overlap=(0.8, 0.8))
#   # detected_boxes = (single_detected_boxes(image, windows_semi))

#   windows_big = slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 700], 
#                       xy_window=(200, 200), xy_overlap=(0.8, 0.8))
#   combined = windows_semi+windows_big
#   detected_boxes = (single_detected_boxes(image, combined))

#   result = duplicate_suppression(image, detected_boxes)
#   return result

# pipeline(image)  

t=time.time()
# ############
# # single image: (comment line 237, 238; uncomment line 239)
windows_medium = slide_window(image, x_start_stop=[200, 1280], y_start_stop=[450, 550], 
                      xy_window=(120, 80), xy_overlap=(0.75, 0.75))

# # detected_boxes = (single_detected_boxes(image, windows_medium))

# # windows_small = slide_window(image, x_start_stop=[500, 1000], y_start_stop=[400, 450], 
# #                     xy_window=(20, 20), xy_overlap=(0.8, 0.8))
# # detected_boxes = (single_detected_boxes(image, windows_small))

windows_small = slide_window(image, x_start_stop=[700, 1100], y_start_stop=[400, 480], 
                    xy_window=(50, 40), xy_overlap=(0.65, 0.65))
# detected_boxes = (single_detected_boxes(image, windows_semi))

windows_big = slide_window(image, x_start_stop=[200, 1280], y_start_stop=[400, 650], 
                    xy_window=(200, 200), xy_overlap=(0.8, 0.8))
combined = windows_medium+windows_big+windows_small
detected_boxes = (single_detected_boxes(image, combined))

window_img = draw_boxes(image, detected_boxes, color=(0, 0, 255), thick=6)


#fill up the intersecting rectangles so to create a full area:
image_copy = np.zeros_like(image)
window_filled = draw_boxes(image_copy, detected_boxes, color=(0, 0, 255), thick=-1)


# hsv = cv2.cvtColor(window_filled, cv2.COLOR_RGB2HSV)
# mask = cv2.inRange(hsv, (110,50,50), (130,255,255))
# # res = cv2.bitwise_and(window_filled,window_filled, mask= mask)
# mask = cv2.erode(mask, None, iterations=2)
# mask = cv2.dilate(mask, None, iterations=2)
from skimage.color import rgb2gray



# heat_map = np.zeros_like(image)

# for (x1, y1, x2, y2) in detected_boxes:
#         heat_map[y1:y2,x1:x2] += 10
        
# print(heat_map[400:480,828:948])

# import heatmap
# hm = heatmap.Heatmap()
# pts =np.array(detected_boxes[0])
# output = hm.heatmap(pts,size=(1280, 720), area = ((0, 0), (1280, 720)))

# overlay = image.copy()
# output = image.copy()
# alpha = 0.6
# for i in detected_boxes:

#   cv2.rectangle(overlay, i[0], i[1],
#       (0, 0, 255), -1)
#   cv2.addWeighted(overlay, alpha, output, 1 - alpha,
#       0, output)

# binary = color_blue_filter(output)
# from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh

# print(heat_map[:,:,0])

# print(image_gray*10)
# # blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.1)

# y, x, r = blob_doh(image_gray, max_sigma= 500, threshold=0.01, min_sigma=0.01)
# print(y[1])
# c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
# x=[]
# for i in detected_boxes:
#   x.append(i[0])
#   x.append(i[2])

# y=[]
# for i in detected_boxes:
#   y.append(i[1],)
#   y.append(i[3])
# print(x)
# print(y)
# heatmap, xedges, yedges = np.histogram2d(x,y, bins=10)
# extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

# #filter for blue color (color of the filled area) and create binary image:
# if detected_boxes != []:
binary = color_blue_filter(window_filled)
ret, thresh = cv2.threshold(binary.astype(np.uint8) * 255, 127, 255, 0)
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# print(contours)
# # #draw rectangle over the detected contour areas and centroid of rectangle:
for contour in contours:
#     #calculate area of contour and remove small areas that could be deceiving:
    # if cv2.contourArea(contour) > 20:
    cnt = contour
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    centroidx = (x+w/2)
    centroidy = (y+h/2)
    cv2.circle(image, (centroidx, centroidy), 10, (0, 255, 0), -1)
    centroid = (centroidx,centroidy)
    print(centroid)
    cv2.drawContours(image, contour, -1, (0,255,0), 3)

plt.imshow(window_img)#, cmap='gray')
t2=time.time()
print(t2-t)
plt.show()

####################


