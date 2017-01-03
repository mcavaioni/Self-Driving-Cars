#Use color transforms, gradients, etc., to create a thresholded binary image.
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

import calibration

path ='/Users/michelecavaioni/Flatiron/My-Projects/Udacity (Self Driving Car)/Project #4 (Advanced Lane Finding)/CarND-Advanced-Lane-Lines/test_images'
images = glob.glob(path + '/*.jpg')

mtx = calibration.mtx
dist = calibration.dist


# print(len(images))
# img = mpimg.imread(images[4])
# gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# undist = cv2.undistort(img, mtx, dist, None, mtx)
# plt.imshow(gray, cmap= 'gray')
# plt.show()

def thres_img(img):
  '''
  combines binary threshold for x gradient and for color S channel
  '''
# for img in images:
  img = mpimg.imread(img)
  #apply distortion correction to the raw image
  cal_img = cv2.undistort(img, mtx, dist, None, mtx)

  #convert to HLS space and separate S channel:
  hls = cv2.cvtColor(cal_img, cv2.COLOR_RGB2HLS)
  s_channel = hls[:,:,2]
  
  #convert to grayscale:
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

  #apply Sobel X filter:
  sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
  abs_sobelX = np.absolute(sobelX)
  scaled_sobel = np.uint8(255*abs_sobelX/np.max(abs_sobelX)) #convert it to 8-bit

  # #apply Sobel Y filter:
  # sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
  # abs_sobelY = np.absolute(sobelY)
  # scaled_sobel = np.uint8(255*abs_sobelY/np.max(abs_sobelY)) #convert it to 8-bit


  #create binary threshold to select pixels based on gradient strength (for x gradient):
  thresh_min = 20
  thresh_max = 100
  sxbinary = np.zeros_like(scaled_sobel)
  sxbinary[(scaled_sobel>=thresh_min) & (scaled_sobel<=thresh_max)] = 1 
  # plt.imshow(sxbinary, cmap = 'gray')
  # plt.show()

  #create binary threshold to select pixels based on gradient strength (for color channel S): 
  s_thresh_min = 175
  s_thresh_max = 255
  s_binary = np.zeros_like(s_channel)
  s_binary[(s_channel>=s_thresh_min) & (s_channel<=s_thresh_max)]=1
  # plt.imshow(s_binary, cmap = 'gray')
  # plt.show()

  #stack each channel to view their individual contributions in green and blue respectively:
  color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))
  # plt.imshow(color_binary)
  # plt.show()

  #combine the two binary thresholds
  combined_binary = np.zeros_like(sxbinary)
  combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
  plt.imshow(combined_binary, cmap='gray')
  plt.show()

