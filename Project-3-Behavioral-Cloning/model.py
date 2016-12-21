# importing all the necessary libraries
import numpy as np
import csv
import os
import sys
import cv2
import pandas as pd
from skimage.io import imread
from sklearn.utils import shuffle
import glob, os
import matplotlib.pyplot as plt
import scipy.misc
from sklearn.cross_validation import train_test_split
from numpy.random import random

def input_img_lab():
    '''
    Function to read the data from the CSV file and create images and labels.
    '''
    #reads the CSV file and names columns 
    raw_data = pd.read_csv("/Users/michelecavaioni/Flatiron/My-Projects/Udacity (Self Driving Car)/Project #3 (Behavioral Cloning)/driving_log.csv", header = 0)
    raw_data.columns = ('Center','Left','Right','Steering_Angle','Throttle','Brake','Speed')

    #adjusting the left camera view, augmenting the steering angle by +0.15
    left = raw_data[['Left', 'Steering_Angle']].copy()
    left.loc[:, 'Steering_Angle'] += 0.15

    #adjusting the right camera view, augmenting the steering angle by -0.15
    right = raw_data[['Right', 'Steering_Angle']].copy()
    right.loc[:, 'Steering_Angle'] -= 0.15

    #creates the path for images and steering angle values
    img_paths = pd.concat([raw_data.Center, left.Left, right.Right]).str.strip()
    angles = pd.concat([raw_data.Steering_Angle, left.Steering_Angle, right.Steering_Angle])

    #converts to  Numpy-array representation
    img_paths = img_paths.as_matrix()
    angles = angles.as_matrix()

    return img_paths,angles

def batch_generator(img_paths, angles, batch_size=128):
    '''
    Function to generate a batch of images and related steering angle to be fed to the "fit_generator" in the model
    '''
    len_data = img_paths.shape[0]
    train_images = np.zeros((batch_size, 32, 64, 3))
    train_steering = np.zeros(batch_size)
    count = None
    while True:
        for j in range(batch_size):
            if count is None or count >= len_data:
                count = 0
            idx = np.random.randint(img_paths.shape[0])
            train_images[j], train_steering[j] = image_modifier(img_paths[idx],angles[idx])
            count += 1
        yield (train_images, train_steering)

def image_modifier(image_path,angle):
    '''
    Function to crop the images and reshaping them to smaller dimension (32x64) and to randomly create a mirror image/steering angle.
    '''
    image = imread(""+image_path)
    image = image[32:135,0:320]
    resized = cv2.resize(image,(64,32))

    #randomly creates mirror images/ateering angle to compensate bias towards left turns.
    flip = np.random.randint(2, size=1)[0]
    if (flip == 1):
        resized = cv2.flip(resized, 1)
        angle *=-1
    return resized,angle



#creates images and steering angle
images,labels = input_img_lab()

#shuffles
images, labels = shuffle(images, labels,random_state=24)

#split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images,labels, test_size=0.10,random_state=42)

########################

#CREATE FINE TUNING AND MODEL:

#import keras libraries
from keras.models import Sequential
from keras.layers import Dense, Activation, ELU, Lambda, LeakyReLU
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import model_from_json
import json
from keras.regularizers import l1, l2
from keras.preprocessing.image import ImageDataGenerator


######

fine_tune = True

#fine tuning: recalling saved model and weigths and fine tune on new images
if fine_tune:
    batch_size =128
    nb_epoch = 5
    learning_rate = 0.001

    print ("### FINE-TUNE mode ###")

    with open('./model.json', 'r') as jfile:
        model = model_from_json(json.load(jfile))

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))

    model.load_weights('./model.h5')

    model.fit_generator(batch_generator(X_train, y_train, batch_size=batch_size),
                            samples_per_epoch=(X_train.shape[0]),
                            nb_epoch=nb_epoch,
                            nb_val_samples = batch_size,
                            validation_data=batch_generator(X_val, y_val, batch_size=batch_size))

    try:
        os.remove("./model.h5")
    except:
        pass

    model.save_weights("./model.h5")

    print("Saved weights to disk")


#creates Convolutional Neural Network
else: 
    #output of the network
    num_classes = 1

    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(32, 64, 3)))
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(LeakyReLU())
    model.add(Convolution2D(32, 3, 3))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(LeakyReLU())
    model.add(Convolution2D(64, 3, 3))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))

    model.summary()

    model.compile(loss='mean_squared_error', optimizer=Adam())
    print(model.summary())


    #create folder
    model_dir = "model_folder/"
    if os.path.exists(model_dir):
      os.rmdir(model_dir)

    os.mkdir(model_dir)

    num_epoch = 50
    batch_size =128
    nb_epoch = 5

    #saves model and weights for each epoch
    for epoch in range(num_epoch):
    #train the model
        model.fit_generator(batch_generator(X_train, y_train, batch_size=batch_size),
                              samples_per_epoch=X_train.shape[0],
                              nb_epoch=nb_epoch,
                              nb_val_samples = batch_size,
                              validation_data=batch_generator(X_val, y_val, batch_size=batch_size))
        #save after each epoch
        json_string = model.to_json()
        with open("{}model_{}.json".format(model_dir, epoch), 'w') as f:
            json.dump(json_string, f)
        model.save_weights("{}model_{}.h5".format(model_dir, epoch))
