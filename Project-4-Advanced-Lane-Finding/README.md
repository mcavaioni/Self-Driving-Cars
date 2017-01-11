
Udacity Self-Driving Car Project 4: Advanced Lane Finding <br>

<b>GOAL:</b> <br>
The goal of this project is to identify road lane lines from a video stream taken from a forward facing camera mounted on a car.

<b>SOLUTION:</b> <br>
The project is developed in several steps; first for a series of images and then, ultimately, on the video stream.

The steps taken are the following: <br>
1- Apply distortion correction to a raw image.<br>
2- Use color transforms, gradients, etc., to create a thresholded binary image.<br>
3- Apply a perspective transform to rectify binary image ("birds-eye view”).<br>
4- Detect lane pixels and fit to find lane boundary.<br>
5- Determine curvature of the lane and vehicle position with respect to center.<br>
6- Warp the detected lane boundaries back onto the original image.<br>
7- Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
8- Video Stream Pipeline<br>

<b>Step 1: Distortion Correction</b><br>
In this step I calibrate the camera using pictures of known shapes (i.e. chessboards) in order to calculate the distortion.
Applying the OpenCV functions “findChessboardCorners()” and “drawChessboardCorners()”
I can detected and sketch the corners on the chessboard images as below.

![camera_calibration_with_corners] (https://github.com/mcavaioni/Self-Driving-Cars/blob/master/Project-4-Advanced-Lane-Finding/output_images/camera_calibration/calibration10_with_corners.jpg)

Next, the location of these corners is used to compute values of “mtx”, “dist”, “rvecs” and “tvecs” using the OpenCV function “calibrateCamera”.

mtx: the camera matrix used to transform 3D points to 2D.
dist: the distortion coefficient
rvecs: rotation vector
tvecs: translation vector

Finally these values are used as inputs in the openCV function “undistort” which provides the image cleared from any distortion induced by the camera lenses.

<center> undist = cv2.undistort(input_img, mtx, dist, None, mtx)</center>

![original_vs_undistorted](https://github.com/mcavaioni/Self-Driving-Cars/blob/master/Project-4-Advanced-Lane-Finding/output_images/camera_calibration/original_vs_undistorted.png)

Here the result of the distortion correction for an image of the road. (the difference is much more subtle than in the previous example. Good reference areas that show the correction are the car’s hood lines)


