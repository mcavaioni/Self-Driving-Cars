#Use color transforms, gradients, etc., to create a thresholded binary image.
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

import calibration

from scipy.signal import find_peaks_cwt

path ='/Users/michelecavaioni/Flatiron/My-Projects/Udacity (Self Driving Car)/Project #4 (Advanced Lane Finding)/CarND-Advanced-Lane-Lines/test_images'
images = glob.glob(path + '/test*.jpg')

mtx = calibration.mtx
dist = calibration.dist


# print(len(images))
# img = mpimg.imread(images[4])
# plt.imshow(img)
# gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# undist = cv2.undistort(img, mtx, dist, None, mtx)
# plt.imshow(gray, cmap= 'gray')
# plt.show()
# print(img.shape)

def thres_img(img):
  '''
  combines binary threshold for x gradient and for color S channel
  '''
  # img = mpimg.imread(img)
  #apply distortion correction to the raw image
  # img = cv2.undistort(img, mtx, dist, None, mtx)

  #convert to HLS space and separate S channel:
  hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
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
  return combined_binary
  # plt.imshow(combined_binary, cmap='gray')
  # plt.show()

def warp(img):
  '''
  selects areas and applies Perspective Transform to get "bird-eye-view"
  '''
  # img = mpimg.imread(img)
  # img = cv2.undistort(img, mtx, dist, None, mtx)
  # plt.imshow(img)
  img_size = (img.shape[1], img.shape[0])

  #trapezoidal
  # plt.plot(200,660, '.')
  # plt.plot(520,480, '.')
  # plt.plot(770,480, '.')
  # plt.plot(1150,660, '.')

  #rectangle
  # plt.plot(0,img.shape[0], '.')
  # plt.plot(0,0, '.')
  # plt.plot(img.shape[1],0, '.')
  # plt.plot(img.shape[1],img.shape[0], '.')

  src = np.float32([[200,660],
                    [520,480],
                    [770,480],
                    [1150,660]])

  dst = np.float32([[0,img.shape[0]],
                    [0,0],
                    [img.shape[1],0],
                    [img.shape[1],img.shape[0]]])

  M = cv2.getPerspectiveTransform(src,dst)
  warped = cv2.warpPerspective(img, M, img_size, flags = cv2.INTER_LINEAR)
  return warped

# for img in images:
img = images[5]
img = mpimg.imread(img)
#apply distortion correction to the raw image
img = cv2.undistort(img, mtx, dist, None, mtx)
thresholded = thres_img(img)
warped_img = warp(thresholded)
# plt.imshow(warped_img, cmap='gray')
histogram = np.sum(warped_img[warped_img.shape[0]/2:,:], axis=0)
# plt.plot(histogram)
# plt.show()

#finding right and left lanes pixels:
def lines_pixels(img):
  right_lane_x = []
  right_lane_y = []
  left_lane_x = []
  left_lane_y = []

  #for left lane:
  hist_left_half = np.sum(img[img.shape[0]/2:,:img.shape[1]/2], axis=0)
  init_peak_left = int(np.argmax(hist_left_half))

  x_val = init_peak_left
  previous = 0
  for i in reversed(range(10)):
    img_y1 = img.shape[0]-img.shape[0]*i/10
    img_y2 = img.shape[0]-img.shape[0]*(i+1)/10

    histogram = np.sum(img[img_y2:img_y1,:img.shape[1]/2], axis=0)
    x_val = int(np.argmax(histogram))
    y_val = int(720-i*img.shape[0]/10)
    if (y_val == 0 or x_val == 0):
      pass
    elif (abs(x_val-previous) > 100 and not(i == 9) and not(previous == 0)):
      pass
    else:
      left_lane_x.append(x_val)
      left_lane_y.append(y_val)
      previous = x_val

  #for right lane:
  hist_right_half = np.sum(img[img.shape[0]/2:,img.shape[1]/2:], axis=0)
  init_peak_right = int(np.argmax(hist_right_half)) + 640

  x_val = init_peak_right
  previous = 0
  for i in reversed(range(10)):
    img_y1 = img.shape[0]-img.shape[0]*i/10
    img_y2 = img.shape[0]-img.shape[0]*(i+1)/10

    histogram = np.sum(img[img_y2:img_y1,(img.shape[1]/2):], axis=0)
    x_val = int(np.argmax(histogram)) + 640
    y_val = int(720-i*img.shape[0]/10)
    if (y_val == 0 or x_val == 640):
      pass
    elif (abs(x_val-previous) > 100 and not(i == 9) and not(previous == 0)):
      pass
    else:
      right_lane_x.append(x_val)
      right_lane_y.append(y_val)
      previous = x_val
          

  plt.imshow(img)
  plt.plot(left_lane_x, left_lane_y, 'o', color='yellow')
  plt.plot(right_lane_x, right_lane_y, 'o', color = 'blue')
  

  #fit polynomial on the left:
  left_lane_y = np.array(left_lane_y)
  left_fit = np.polyfit(left_lane_y, left_lane_x, 2)
  left_fitx = left_fit[0]*left_lane_y**2 + left_fit[1]*left_lane_y + left_fit[2]
  plt.plot(left_fitx, left_lane_y, color='green', linewidth=3)

  #fit polynomial on the right:
  right_lane_y = np.array(right_lane_y)
  right_fit = np.polyfit(right_lane_y, right_lane_x, 2)
  right_fitx = right_fit[0]*right_lane_y**2 + right_fit[1]*right_lane_y + right_fit[2]
  plt.plot(right_fitx, right_lane_y, color='green', linewidth=3)
  plt.show()

lines_pixels(warped_img)
# plt.show()

