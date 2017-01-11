
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
