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


**Step 1: Histogram of Color Features**
A simple template matching approach quickly shows its limitations when the object we are detecting changes in size, orientation or color.
A more robust transformation is the Histogram of Color Features, which, as the name suggests, computes the histogram of color values in an image and is not sensitive to a perfect arrangement of pixels. 
Below the result of Histogram of Color to the original image.
*[Ref. Lines 26 - 37 in pipeline.py]*


![original](https://github.com/mcavaioni/Self-Driving-Cars/blob/master/Project-5-Vehicle-Detection-and-Tracking/output_images/image0027.png)
<p style="text-align: center;"> Original image </p>

![histogram of color](https://github.com/mcavaioni/Self-Driving-Cars/blob/master/Project-5-Vehicle-Detection-and-Tracking/output_images/histogram_of_color_image0027.png)
<p style="text-align: center;"> Histogram of colors </p>

**Step 2: Binned Spatial Features**
As Mentioned above, template matching with raw pixel values is not a robust method, but it still retains good information that we want to use. Although including three color channels of a full resolution image is a bit extensive, so we can perform spatial binning, reducing the dimensions and resolution of an image, while still retaining enough information to help in finding vehicles.
In my case, I resize the images to a 32x32 spatial space and then use “.ravel()” to create the feature vector.
*[Ref. Lines 41 - 48 in pipeline.py]*

![binned spatial image](https://github.com/mcavaioni/Self-Driving-Cars/blob/master/Project-5-Vehicle-Detection-and-Tracking/output_images/binned_spatial_image0027.png)
<p style="text-align: center;"> Spatial binned features in RGB space </p>

**Step 3: Histogram of Oriented Gradients (HOG) Features**
So far we still haven’t captured an important factor, which is the notion of shape.
Gradient values provide that signature, but they make the signature far too sensitive.
The HOG method helps in depicting the gradient, its magnitude and direction.
These three characteristics are captured for each pixel, but then they are grouped together in small cells and finally for each cell we compute the histogram of the gradient directions.
This is finally “block normalized” across blocks of cells to create better invariance to illumination, shadow and edge contrast. Finally we flatten these features into a feature vector.
I have tested this method using different parameters, such as orientation, pix_per_cell and cell_per_block. I achieved acceptable results reducing the number of pix_per_cell to 4x4 but the computing time was very high. I finally settled with the following parameters that provided a good computing time and great results:
orientation:9; pix_per_cell: 8x8; cell_per_block: 2
*[Ref. Lines 50 - 66 in pipeline.py]*

![HOG](https://github.com/mcavaioni/Self-Driving-Cars/blob/master/Project-5-Vehicle-Detection-and-Tracking/output_images/HOG_GTI_right_image0027.png)
<p style="text-align: center;"> Original iamge and HOG features </p>

**Step 4: Color Space Features**
I initially used the “regular” RGB color space but a better improvement was achieved transforming the image from RGB space to YCrCb space as well.
[Ref. Line 93 in pipeline.py]
The HOG features method was applied to all three channels of the selected color space.
*[Ref. Lines 100 - 105 in pipeline.py]*

![color space](https://github.com/mcavaioni/Self-Driving-Cars/blob/master/Project-5-Vehicle-Detection-and-Tracking/output_images/color_space_ycrcb.png)
<p style="text-align: center;"> Color space YCbCr features </p>

**Step 5: Combine features**
I decided to use a combination of the above mentioned features, gathering more useful information about each image.
*[Ref method “extract_features”: lines 69 - 109 in pipeline.py]*
The combination of this features gets applied for each set of car and non-car images.
Finally, to avoid different magnitude among the combined features I normalize them.
*[Ref. Lines 127 - 131 in pipeline.py]*

![combined](https://github.com/mcavaioni/Self-Driving-Cars/blob/master/Project-5-Vehicle-Detection-and-Tracking/output_images/combined_features.png)
<p style="text-align: center;"> Original imag, raw combined features and normalized </p>

**Step 6: Train a Classifier**
Before fitting the features into the classifier I created a label vector for the “car” and “non-car” images, classified as 1 and 0 respectively.
After that, the data was randomly split into training and test data.
I tested several classifiers, such as MLP, SVC and finally I used the LinearSVC classifier as the result were good and the training speed was very reasonable compared to te previous ones.
I tweaked a little bit some parameters within the LinearSVC classifier, in particular the “C” parameter, increasing it from the standard value of 1.0 to 100.
In fact, C controls the tradeoff between smooth decision boundary and classifying training points correctly. A high value of C gets more training points correctly, giving a more “intricate” decision boundary (when looking at a scattered plot diagram). The pitfall is to overfit the data. 
Here below the values achieved with the LinearSVC(penalty='l1',dual=False, C=100) classifier:

(225.28446888923645, 'Seconds to train SVC...')
('Train Accuracy of SVC = ', 1.0)
('Test Accuracy of SVC = ', 0.99209183673469392)
(0.00043582916259765625, 'Seconds to predict with SVC')


**Step 7: Sliding Window Search and Vehicle Detection**
After training the classifier we want to use it to predict an object, classifying it. In our case we want to map the whole image dividing it in subregions and run the classifier on each one of them. 
To do that I created different window sizes (small, medium, big) which slide through the image in different locations. I restricted the search from a y_value of 400 and up, since this section of the image is only including the road and not anything above it. Also, each window size maps a specific region, assigning to them different “x_start_stop”, “y_start_stop” values. Finally the sliding window is achieved with different overlaps across each subsequent window.
*[Ref “slide_window” method in lines 127 - 159 and lines 215 - 222 for specific window sizes (in pipeline.py) ]*
Here below is the combination all the window sizes mapped throughout the image section. 

![sliding window](https://github.com/mcavaioni/Self-Driving-Cars/blob/master/Project-5-Vehicle-Detection-and-Tracking/output_images/window_sliding.png)
