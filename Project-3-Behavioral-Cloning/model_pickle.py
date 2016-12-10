import glob, os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

path = '/Users/michelecavaioni/Flatiron/My-Projects/Udacity (Self Driving Car)/Project #3 (Behavioral Cloning)/IMG/'

#images in list as center+left+right
image_dict = []
for i,infile in enumerate(glob.glob(os.path.join(path,'*.jpg'))):
    img = cv2.imread(infile)   
    resized = scipy.misc.imresize(img, (100,200))      
    # image_dict.append(img)
    image_dict.append(resized)

image_dict = np.array(image_dict) 


# print(len(image_dict))
#(8685)
# print(image_dict.shape)
#(8685, 100, 200, 3) or if not resized:(8685, 160, 320, 3)

#to show image:
# plt.imshow(image_dict[0])
# plt.show()

import pandas as pd
csv_file = '/Users/michelecavaioni/Flatiron/My-Projects/Udacity (Self Driving Car)/Project #3 (Behavioral Cloning)/driving_log_edit.csv'
df = pd.read_csv(csv_file)
steering_angle = df['Steering Angle']
steering_angle = steering_angle.values.tolist()

# print(len(steering_angle))
#2895

#modify steering angle from the center value in left images:
steering_angle_with_left=[]
for i in steering_angle:
  #if turning left (<0 value) => make softer turn so add value (negative becomes less negative; positive(right turn becomes bigger, harder turn))
  if i == 0:
    steering_angle_with_left.append(i)
  else:
    left_img_i = i + 0.08
    steering_angle_with_left.append(left_img_i)

#modify steering angle from the center value in right images:
steering_angle_with_right=[]
for i in steering_angle:
  #subtract vaue to make smaller right turn value and more negative value for left turn:
  if i == 0:
    steering_angle_with_right.append(i)
  else:
    right_img_i = i - 0.08
    steering_angle_with_right.append(right_img_i)

#steering angle values as center+left+right:
steering_angle = np.array(steering_angle + steering_angle_with_right + steering_angle_with_left)

print(len(steering_angle))

import pickle 
with open('/Users/michelecavaioni/Flatiron/My-Projects/Udacity (Self Driving Car)/Project #3 (Behavioral Cloning)/training_data', 'wb') as f:
    var = {'features' : image_dict , 'labels' : steering_angle}
    pickle.dump(var, f)