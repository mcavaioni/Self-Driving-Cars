import numpy as np
import pickle
import glob, os
import cv2 
import matplotlib.image as mpimg
import scipy.misc

#non-vehicles
path = '/Users/michelecavaioni/Flatiron/My-Projects/Udacity (Self Driving Car)/Project #5 (Vehicle Detection and Tracking/images/non_vehicle'

#create array of non-vehicle images
non_vehicle_img = []
for i,infile in enumerate(glob.glob(os.path.join(path,'*.png'))):
    img = scipy.misc.imread(infile)         
    non_vehicle_img.append(img)
non_vehicle_img = np.array(non_vehicle_img)

#pickle non-vehicle images
with open('non_vehicle4.pickle', 'wb') as handle:
    pickle.dump(non_vehicle_img, handle, protocol=pickle.HIGHEST_PROTOCOL)


#only cars:
path_car = '/Users/michelecavaioni/Flatiron/My-Projects/Udacity (Self Driving Car)/Project #5 (Vehicle Detection and Tracking/images/vehicles'

#create arrays of car images from different folders
car_img_gti_far = []
for i,infile in enumerate(glob.glob(os.path.join(path_car,'GTI_Far/*.png'))):
    img = scipy.misc.imread(infile)          
    car_img_gti_far.append(img)
car_img_gti_far = np.array(car_img_gti_far)

car_img_gti_left = []
for i,infile in enumerate(glob.glob(os.path.join(path_car,'GTI_Left/*.png'))):
    img = scipy.misc.imread(infile)          
    car_img_gti_left.append(img)
car_img_gti_left = np.array(car_img_gti_left)

car_img_gti_right = []
for i,infile in enumerate(glob.glob(os.path.join(path_car,'GTI_Right/*.png'))):
    img = scipy.misc.imread(infile)            
    car_img_gti_right.append(img)
car_img_gti_right = np.array(car_img_gti_right)

car_img_gti_middle = []
for i,infile in enumerate(glob.glob(os.path.join(path_car,'GTI_MiddleClose/*.png'))):
    img = scipy.misc.imread(infile)          
    car_img_gti_middle.append(img)
car_img_gti_middle = np.array(car_img_gti_middle)

car_img_gti_kitti= []
for i,infile in enumerate(glob.glob(os.path.join(path_car,'KITTI_extracted/*.png'))):
    img = scipy.misc.imread(infile)           
    car_img_gti_kitti.append(img)
car_img_gti_kitti = np.array(car_img_gti_kitti)

#combine all the images from all the arrays taken from different folders
car_img = np.concatenate((car_img_gti_far, car_img_gti_left, car_img_gti_right, car_img_gti_middle, car_img_gti_kitti))


#pickle car images
with open('car4.pickle', 'wb') as handle:
    pickle.dump(car_img, handle, protocol=pickle.HIGHEST_PROTOCOL)


# #open file:
# fileObject = open(file_Name,'r')  
# # load the object from the file into var b
# b = pickle.load(fileObject) 

#better:
# with open('non_veh.pickle', 'rb') as handle:
#     b = pickle.load(handle)

