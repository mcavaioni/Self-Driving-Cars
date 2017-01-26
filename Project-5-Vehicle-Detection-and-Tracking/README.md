<b>GOAL:</b></br>
The goal of this project is to write a software pipeline to identify vehicles in a video from a front-facing camera on a car.

<b>SOLUTION:</b></br>
The pipeline is developed in several steps; first for images and then, ultimately, on the video stream.

The steps taken are the following:
1- Compute histogram of color features 
2- Compute binned spatial features 
3- Compute Histogram of Oriented Gradients (HOG) features
4- Compute color space features
5- Combine the above features into a list of feature vectors
6- Train a classifier
7- Create a sliding window search and detect vehicles in an image
8- Reducing false positives
9- Draw bounding boxes on each overlapping detection 
10- Video Stream Pipeline 
11- Reduce/segregate false positive in a Video Stream 


**Step 1: Histogram of Color Features**</br>
A simple template matching approach quickly shows its limitations when the object we are detecting changes in size, orientation or color.
A more robust transformation is the Histogram of Color Features, which, as the name suggests, computes the histogram of color values in an image and is not sensitive to a perfect arrangement of pixels. 
Below the result of Histogram of Color to the original image.</br>
*[Ref. Lines 26 - 37 in pipeline.py]*


![original](https://github.com/mcavaioni/Self-Driving-Cars/blob/master/Project-5-Vehicle-Detection-and-Tracking/output_images/image0027.png)
<p style="text-align: center;"> Original image </p>

![histogram of color](https://github.com/mcavaioni/Self-Driving-Cars/blob/master/Project-5-Vehicle-Detection-and-Tracking/output_images/histogram_of_color_image0027.png)
<p style="text-align: center;"> Histogram of colors </p>

**Step 2: Binned Spatial Features**</br>
As Mentioned above, template matching with raw pixel values is not a robust method, but it still retains good information that we want to use. Although including three color channels of a full resolution image is a bit extensive, so we can perform spatial binning, reducing the dimensions and resolution of an image, while still retaining enough information to help in finding vehicles.
In my case, I resize the images to a 32x32 spatial space and then use “.ravel()” to create the feature vector.</br>
*[Ref. Lines 41 - 48 in pipeline.py]*

![binned spatial image](https://github.com/mcavaioni/Self-Driving-Cars/blob/master/Project-5-Vehicle-Detection-and-Tracking/output_images/binned_spatial_image0027.png)
<p style="text-align: center;"> Spatial binned features in RGB space </p>

**Step 3: Histogram of Oriented Gradients (HOG) Features**</br>
So far we still haven’t captured an important factor, which is the notion of shape.
Gradient values provide that signature, but they make the signature far too sensitive.
The HOG method helps in depicting the gradient, its magnitude and direction.
These three characteristics are captured for each pixel, but then they are grouped together in small cells and finally for each cell we compute the histogram of the gradient directions.
This is finally “block normalized” across blocks of cells to create better invariance to illumination, shadow and edge contrast. Finally we flatten these features into a feature vector.
I have tested this method using different parameters, such as orientation, pix_per_cell and cell_per_block. I achieved acceptable results reducing the number of pix_per_cell to 4x4 but the computing time was very high. I finally settled with the following parameters that provided a good computing time and great results:
orientation:9; pix_per_cell: 8x8; cell_per_block: 2</br>
*[Ref. Lines 50 - 66 in pipeline.py]*

![HOG](https://github.com/mcavaioni/Self-Driving-Cars/blob/master/Project-5-Vehicle-Detection-and-Tracking/output_images/HOG_GTI_right_image0027.png)
<p style="text-align: center;"> Original iamge and HOG features </p>

**Step 4: Color Space Features**</br>
I initially used the “regular” RGB color space but a better improvement was achieved transforming the image from RGB space to YCrCb space as well.
[Ref. Line 93 in pipeline.py]
The HOG features method was applied to all three channels of the selected color space.</br>
*[Ref. Lines 100 - 105 in pipeline.py]*

![color space](https://github.com/mcavaioni/Self-Driving-Cars/blob/master/Project-5-Vehicle-Detection-and-Tracking/output_images/color_space_ycrcb.png)
<p style="text-align: center;"> Color space YCbCr features </p>

**Step 5: Combine features**</br>
I decided to use a combination of the above mentioned features, gathering more useful information about each image.
*[Ref method “extract_features”: lines 69 - 109 in pipeline.py]*
The combination of this features gets applied for each set of car and non-car images.
Finally, to avoid different magnitude among the combined features I normalize them.</br>
*[Ref. Lines 127 - 131 in pipeline.py]*

![combined](https://github.com/mcavaioni/Self-Driving-Cars/blob/master/Project-5-Vehicle-Detection-and-Tracking/output_images/combined_features.png)
<p style="text-align: center;"> Original imag, raw combined features and normalized </p>

**Step 6: Train a Classifier**</br>
Before fitting the features into the classifier I created a label vector for the “car” and “non-car” images, classified as 1 and 0 respectively.
After that, the data was randomly split into training and test data.
I tested several classifiers, such as MLP, SVC and finally I used the LinearSVC classifier as the result were good and the training speed was very reasonable compared to te previous ones.
I tweaked a little bit some parameters within the LinearSVC classifier, in particular the “C” parameter, increasing it from the standard value of 1.0 to 100.
In fact, C controls the tradeoff between smooth decision boundary and classifying training points correctly. A high value of C gets more training points correctly, giving a more “intricate” decision boundary (when looking at a scattered plot diagram). The pitfall is to overfit the data. 
Here below the values achieved with the LinearSVC(penalty='l1',dual=False, C=100) classifier:</br>

(225.28446888923645, 'Seconds to train SVC...')</br>
('Train Accuracy of SVC = ', 1.0)</br>
('Test Accuracy of SVC = ', 0.99209183673469392)</br>
(0.00043582916259765625, 'Seconds to predict with SVC')</br>


**Step 7: Sliding Window Search and Vehicle Detection**</br>
After training the classifier we want to use it to predict an object, classifying it. In our case we want to map the whole image dividing it in subregions and run the classifier on each one of them. 
To do that I created different window sizes (small, medium, big) which slide through the image in different locations. I restricted the search from a y_value of 400 and up, since this section of the image is only including the road and not anything above it. Also, each window size maps a specific region, assigning to them different “x_start_stop”, “y_start_stop” values. Finally the sliding window is achieved with different overlaps across each subsequent window.</br>
*[Ref “slide_window” method in lines 127 - 159 and lines 215 - 222 for specific window sizes (in pipeline.py) ]*</br>
Here below is the combination all the window sizes mapped throughout the image section. 

![sliding window](https://github.com/mcavaioni/Self-Driving-Cars/blob/master/Project-5-Vehicle-Detection-and-Tracking/output_images/window_sliding.png)

Each window is then resized to a 64x64 before extracting the combined features as described above.
I then normalize the feature vector using the scaler saved from the classifier in order to normalize it at the same level as the trained features.
The saved classifier is finally run on this normalized feature in order to predict if the subregion of the image is a car or not.
The results that I achieved were good, although some false positives were still getting detected. </br>
*[Ref. “single_detected_boxes” method in Lines 163 - 194 in pipeline.py]*</br>
I tackled this issues with two approaches:
- thresholding the prediction function</br>
- hard-negative-mining



**Step 8: Reducing False Positives**</br>
The first approach was to use the “decision_function” from the classifier, which predicts the confidence scores. This score is not intuitive as the probability score would be, but looking at different case scenarios I decided to set the threshold to a value of 1.0, which was filtering most of the “weak” false positives. I didn’t want to set this threshold too high as in some cases it was neglecting also true positives.
Therefore, I took a second action towards the false positive reduction, using “hard-negative-mining”.
This method reveals to be very “manual” and time consuming as it entails to crop the position of the image that is classified incorrectly and use it as a “non-vehicle” sample in the training images.
I ended up cropping around 100 images and then I retrained the classifier and eventually implemented this method a couple of times.
The improvements were very substantial. 
Here below the final result of vehicle detection using sliding window search, after thresholding and hard-negative-mining:

![detected boxes](https://github.com/mcavaioni/Self-Driving-Cars/blob/master/Project-5-Vehicle-Detection-and-Tracking/output_images/detected%20boxes.png)


**Step 9: Draw Bounding Boxes**</br>
As we see from the image above the car gets detected several times by several overlapping windows of different sizes. 
What we finally want to represent is a bounding box around the overall detection.
There are several approaches that could be used. I tried “watershed segmentation” from the scikit-image library, but I ultimately ended up using an approach similar to it.
I filled all the detected boxes in a blue color and then applied a filter to the image in order to detect the whole filled area, finally transforming it into a binary image.</br>
*[Ref. “color_blue_filter” method Lines 198 - 207 in pipeline.py]*

Then, I thresholded the binary image and detected the contours around it.</br>
*[Ref. Lines 242 - 243 in pipeline.py]*

Here below the final result with the detected boxes in blue color and the final surrounding contour in green color.

![detected boxes with contour](https://github.com/mcavaioni/Self-Driving-Cars/blob/master/Project-5-Vehicle-Detection-and-Tracking/output_images/detected_boxes_with_contour.png)


**Step 10: Video Stream Pipeline**</br>
The video stream pipeline is constructed similar to the image pipeline, although it implements a further method in order to additionally segregate false and true positives.
First, in each frame I implemented the search as described above (sliding-window search plus classifier) and in each frame the detected vehicle positions is drawn with a grey box.
As we can see from the video there are still some false positives depicted.
In order to further reduce them, the video stream pipeline implements the following approach.
For every 7 frames in the video stream the prior 6 detected bounding boxes are collected and overlapped with each other.
Each overlapping area produces a heat map which is thresholded to a certain value that discerns the bounding boxes that appear in a near position (therefore overlapping) for most of the times, meaning for most of the prior 6 detected frames.</br>
*[Ref. Lines 288 - 294 in pipeline.py]*</br>
These high-confidence detections where multiple overlapping detections occur are then identified with the method described above (finding the surrounding contours) and ultimately drawn in a green color, together with the centroid of this green bounding box.

Finally the video is combined with the advanced lane finding pipeline implemented in the previous project.

**IMPROVEMENTS:**</br>
Currently the pipeline runs a frame every approximately 2 seconds. I reduced an initial higher value (~3.5 seconds) selecting different values for the window search parameters in order to limit the number of windows where the classifier has to run.
This result is still not industry acceptable, being far from a real time implementation.
Improving this speed will also provide smoother transition in the video pipeline, showing the green boxes in a closer time lap.
Some of the recurrent false positives could still be filtered out applying further hard-negative-mining, but I decided not to do that in order to not make the pipeline “just right” for this video stream.
Furthermore with a more real-time implementation I could further enhance the pipeline using the detected centroid to track the vehicles’ position.

