
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

<p align= "center"> undist = cv2.undistort(input_img, mtx, dist, None, mtx)</p>

![original_vs_undistorted](https://github.com/mcavaioni/Self-Driving-Cars/blob/master/Project-4-Advanced-Lane-Finding/output_images/camera_calibration/original_vs_undistorted.png)

Here the result of the distortion correction for an image of the road. (the difference is much more subtle than in the previous example. Good reference areas that show the correction are the car’s hood lines)

![orig_vs_undist_car](https://github.com/mcavaioni/Self-Driving-Cars/blob/master/Project-4-Advanced-Lane-Finding/output_images/original_vs_undistorted_car.png)


<b>Step 2: Thresholded Binary Image</b><br>
In this step I take the undistorted image as an input and apply several “filters” to transform the color image in a binary image in order to highlight pixels in lane lines.
I practiced several gradients and color transforms, getting mixed results.
Ultimately I settled with a combination of gradient direction threshold and two color transforms.

The direction of the gradient is simply the arctangent of the y-gradient divided by the x-gradient (tan^-1(sobley/sobelx)). Each pixel of the resulting image contains a value for the angle of the gradient away from horizontal in units of radians. This threshold by itself provides a very noisy image, but the combination with the following transforms highlights the lanes lines pretty clearly.

The first color transform I used is detecting the “L” channel from a “LUV” field. Playing with different values I was able to use this color channel to best highlight the white lanes.

The second color transform I used is detecting the “B” channel from a “LAB” field. This works best, instead, on the detection of the yellow lanes, completely ignoring instead the white ones.

Finally I combined the gradient and the color transforms and obtained a binary image where the lane lines where mostly depicted.

![thresholded](https://github.com/mcavaioni/Self-Driving-Cars/blob/master/Project-4-Advanced-Lane-Finding/output_images/test1_thresholded.jpg)

<b>Step 3: Perspective Transform (“birds-eye view”)</b><br>
In this step I transformed the previous binary image seen from the front camera with its perspective into a “birds-eye-view” image of the road.
The objects in a “regular”  2D image appear smaller the greatest their distance from the camera (the distance is the “z value” in an image).
The perspective transform converts the “z” coordinate of the object points, “dragging” points towards or “pushing” them away from the camera, in order to change the apparent perspective.
In our case applying the perspective transform to a road image allows to see the road from above in such a way that the road lines appear parallel to each other (as it would be if the image was taken from above instead of from a front-facing camera).
To obtain that I apply the OpenCV function “getPerspectiveTransform” which takes four source points (“src”) from the initial image and four destination points (“dst”) on the warped (“transformed”) image.
The “src” points were manually selected on the binary image in order to locate the position of the lane lines. They form a polygon which “follows” the lane lines.
The “dst” points were also manually selected in order to form a rectangle in the final, warped image.
Finally, the result of the “getPerspectiveTransform” function is a matrix which is used as an input in the “warpPerspective” function which ultimately transforms the image.

![warped](https://github.com/mcavaioni/Self-Driving-Cars/blob/master/Project-4-Advanced-Lane-Finding/output_images/test1_warped.jpg)


<b>Step 4: Detect Lane Pixels and fit Polynomial to find lane boundary</b><br>
For this step I first divide the image in left side and right side in order to separate the left lane from the right one.
Then for each side I horizontally divide the image in two parts and I take a histogram along all the columns in the lower half of the image like this (for the left side):

<p align= "center"> hist_left_half = np.sum(img[img.shape[0]/2:,:img.shape[1]/2], axis=0) </p>

With this histogram I am adding up the pixel values along each column in the half image. In my thresholded binary image, pixels are either 0 or 1, so the two most prominent peak in this histogram will be good indicator of the x-position of the base of the lane line. I can use that as a starting point for where to search for the lines. I set this value as the initial_peak_value.

<p align= "center"> init_peak_left = int(np.argmax(hist_left_half)) </p>

From this point I divide the image (left side and right side) in different horizontal sections and for each section I apply the histogram peaks search, but evaluating only the peaks that are in closed proximity with the previous section peak value (starting with the initial_peak_value).
For “closed proximity” I set a value of 100 as a range from previous detected peak and current peak. 
This allows to discard peaks (therefore pixels) that are far off from the lane line initially identified throughout the histogram search on half an image. 
The detected peaks (x and y pixel values) are collected in an array.

Finally I fit a second order polynomial (f(y) = A*y^2+B*y+C) to the detected pixels position for the right and left lane lines:

<p align= "center"> left_fit = np.polyfit(left_lane_y, left_lane_x, 2) </p>
<p align= "center"> left_fitx = left_fit[0]*left_lane_y**2 + left_fit[1]*left_lane_y + left_fit[2] </p>

![warped_with_lines](https://github.com/mcavaioni/Self-Driving-Cars/blob/master/Project-4-Advanced-Lane-Finding/output_images/test1_warped_with_lines.jpg)

<b>Step 5: Curvature of the Lane and Vehicle Position</b><br>
The position of the car with respect to the center of the lane lines is calculated at a y-value at the bottom position of the image. I calculate the intercept between a horizontal line at that y-value and the two polynomials (right and left). The center of the lanes is the middle point of the two intercepts and the horizontal position of the car is the middle vertical point of the image (image_width/2 = 640).
The distance from the center is the difference between the car position and the middle_lane (center between the two lanes).
This value is then transformed in meter multiplying it by (3.7/700).
The car is left from the center line if (640-middle_lane)*(3.7/700) < 0.

The radius of the curvature is calculated with respect to the max y-value (therefore towards the bottom of the image) and it’s evaluated for both lane lines (right and left).
Some difference in values is evident between the two curvatures, which is obviously not so in reality.
The problem relies on the fact that the left lane is a continuous lane and the pixels detected are numerous. On the other hand, the right lane is a dotted line and the pixels detected are quite few in several frames. Therefore the polynomial fitted on the left lane is more accurate (more representative of the entire line) than the one on the right side.
Hence, for my final road curvature value I decided to take the radius calculated on the left side, instead of averaging the two values.

<b>Step 6: Warp the detected lane boundaries back onto the original image</b><br>
The final computed step is used to re-transform the detected lines back into the original image.
To do so, I use the prior OpenCV functions (“getPerspectiveTransform” and “warpPerspective”) where in this case the first function calculates the inverse matrix from “dst” points to “src” points and the “warpPerspective” function computes the final transformation.
Prior to that I extend the polynomial lanes towards the bottom edge of the image and fill the space between the right and left polynomial to highlight the lane area that the car is in.


<b>Step 7: Output Visual Display</b><br>
This step is the visual representation of the previous step, in which the image is the original one in which the detected lane is identified and the values of curvature and car position are revealed.

![original_with_area](https://github.com/mcavaioni/Self-Driving-Cars/blob/master/Project-4-Advanced-Lane-Finding/output_images/test1_original_with_area.jpg)

<b>Step 8: Video Stream Pipeline</b><br>
The approach to the video stream analysis is the same as the one described above for the images, although some peculiarities are introduced in order to identify the lane lines in different frames of the video in order to gain confidence and speed up the computational steps.
In particular, difference line detection approaches are taken for:
first video frame
following 10 video frames
remaining video frames
In the last two cases I also added an exception in the occurrence it fails to detect the lines with the “histogram search”, rolling it back to values of the previous frame.
I created a Line Class which collects information about lines from frame to frame.

*First video frame:*<br>
In the first video frame I calculate the pixels and polynomial values with the “histogram search” method, as described in Step 4.
I also create the first instances of left and right Lines Class, which are used as “prior” values for the subsequent frame.

*Following 10 video frames:*<br>
For each of the next ten video frames I calculate the pixels and polynomial values as in Step 4 and I keep track of “line detection” with the following methodology.
In each frame the values of the curvature of the lane line (A value in the f(y)=A*y^2+B*y+C formula, or the fit[0] value in the code) and the heading or direction that the line is pointing to (B value in the formula, or fit[1] in the code) are compared with the values of the previous frame.
If A and B are differing from the previous ones by a 10%<sup>*</sup> range than I consider the line as “detected” and I keep track of it modifying the property of Line Class instance (self.detected) to True.
If for all the ten frames the lines (left and right) are detected every time, I consider it a “high confidence” detection and use the last frame line detection as the starting point for the subsequent frame.

( <sup>*</sup> the 10% range was a reasonable value based on experience of several video frames)

*Remaining video frames:*<br>
For these final video frames I adopted an if/else approach.
The first approach evaluates the previous “high confidence” result from the batch of the first ten frames (which is mainly used as a consideration for the first frame of this remaining batch) in addition with a check on the pixel detection from a previous frame.
If the above statements are true, meaning the “high confidence” level is reached and the previous frame detected some pixels, then for the current frame I use a faster pixel detection method, which evaluates the “nonzero” pixels within an image window which is drawn around the points detected in the previous frame.

The second approach instead is implemented in the case of no “high-confidence” level and no pixels detected in the previous frame. In this situation, I reuse the “histogram search” pixels detection methodology, re-evaluating from scratch the entire image frame.

Finally, for each frame I depict the values of curvature and car position with respect to the center line (as explained in Step 5).

<b>IMPROVEMENTS:</b><br>
The current pipeline works well for the given video, which has clear image frames with low level of disturbances and overall image noise.
Also, currently in case the lines are not detected with the “histogram search” method, I roll-back to the values calculated in the previous frame. This could be a bit problematic if it occurs for many frames in a row, creating unreliable lines as they carry the previous detected values.
One idea would be to implement a deep learning CNN to predict the left/right lane polynomial coefficients in case of failure of detecting the current lines.

______________

<i>Files used:</i><br>
<b>calibration.py</b> - The script used to calibrate the camera and obtain mtx and dist values. <br>
<b>pipeline.py</b> - The script used to create pipeline for images and video stream. </br>
