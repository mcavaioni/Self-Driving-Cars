import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

path ='/Users/michelecavaioni/Flatiron/My-Projects/Udacity (Self Driving Car)/Project #4 (Advanced Lane Finding)/CarND-Advanced-Lane-Lines/'
images = glob.glob(path + 'camera_cal/calibration*.jpg')


nx = 9
ny = 6

objpoints = []
imgpoints = []

#prepare object points (like (0,0,0), (1,0,0),... (7,5,0)..)
objp = np.zeros((ny*nx,3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

for fname in images:
  img = mpimg.imread(fname)
  #transform in gray image
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  #find chessboard corners
  ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
  #if corners are found, add object points and image points:
  if ret == True:
    imgpoints.append(corners)
    objpoints.append(objp)
    img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
    plt.imshow(img)

#calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


