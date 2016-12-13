# import pickle
# import numpy as np

# with open('/Users/michelecavaioni/Flatiron/My-Projects/Udacity (Self Driving Car)/Project #3 (Behavioral Cloning)/training_data','rb') as f:
#     var = pickle.load(f)

# images = var['features']
# steering_angle = var['labels']

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
    resized = scipy.misc.imresize(img, (20,64))      
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
# steering_angle = steering_angle.values.tolist()

# # print(len(steering_angle))
# #2895

# #modify steering angle from the center value in left images:
# steering_angle_with_left=[]
# for i in steering_angle:
#   #if turning left (<0 value) => make softer turn so add value (negative becomes less negative; positive(right turn becomes bigger, harder turn))
#   if i == 0 or i>0:
#     steering_angle_with_left.append(i)
#   else:
#     left_img_i = i + 0.08
#     steering_angle_with_left.append(left_img_i)

# #modify steering angle from the center value in right images:
# steering_angle_with_right=[]
# for i in steering_angle:
#   #subtract vaue to make smaller right turn value and more negative value for left turn:
#   if i == 0 or i<0:
#     steering_angle_with_right.append(i)
#   else:
#     right_img_i = i - 0.08
#     steering_angle_with_right.append(right_img_i)

# #steering angle values as center+left+right:
# steering_angle = np.array(steering_angle  + steering_angle_with_left + steering_angle_with_right)
steering_angle = np.concatenate((steering_angle, (steering_angle - .12), (steering_angle + .12)))

images = image_dict
steering_angle = steering_angle.astype(np.float32)

# Create a mirror image of the images in the dataset to combat left turn bias
mirror = np.ndarray(shape=(images.shape))
count = 0
for i in range(len(images)):
    mirror[count] = np.fliplr(images[i])
    count += 1
mirror.shape

# Create mirror image labels
mirror_angles = steering_angle * -1





# Combine regular features/labels with mirror features/labels
images = np.concatenate((images, mirror), axis=0)
steering_angle = np.concatenate((steering_angle, mirror_angles),axis=0)

from skimage.exposure import adjust_gamma
images = adjust_gamma(images)

from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
X_train, X_val, y_train, y_val = train_test_split(images, steering_angle, test_size=0.33, random_state=42)

# y_train = np_utils.to_categorical(y_train)
# y_val = np_utils.to_categorical(y_val)
# num_classes = y_val.shape[1]
num_classes = 1

# from keras.preprocessing.image import ImageDataGenerator


def myGenerator():
  while 1:
    for i in range(0, len(X_train)-1 ):
      yield (X_train[i*32:(i+1)*32], y_train[i*32:(i+1)*32])

def myValGenerator():
  while 1:
    for i in range(0, len(X_val)-1 ):
      yield (X_val[i*32:(i+1)*32], y_val[i*32:(i+1)*32])

from keras.models import Sequential
from keras.layers import Dense, Activation, ELU, Lambda, SpatialDropout2D
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
# num_filters1 = 24
# filter_size1 = 5
# stride1=(2,2)
# num_filters2 = 36
# filter_size2 = 5
# stride2=(2,2)
# num_filters3 = 48
# filter_size3 = 5
# stride3=(2,2)
# num_filters4 = 64
# filter_size4 = 3
# stride4=(1,1)
# num_filters5 = 64
# filter_size5 = 3
# stride5=(1,1)
# pool_size = (2, 2)
# hidden_layers1 = 100
# hidden_layers2 = 50
fine_tune_mode = False

# Load the existing model  & weights if we are fine tuning
if fine_tune_mode:
    learning_rate = 0.00001
    print("Running in FINE-TUNE mode!")
    with open("model.json", 'r') as jfile:
        model = model_from_json(json.load(jfile))
    adam = Adam(lr=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=adam)
    weights_file = "model.h5"
    model.load_weights(weights_file)
    print("loaded model from disk")
    model.summary()
else:
  # Otherwise build a new CNN Network with Keras
    learning_rate = 0.0001  #0.0001

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(20, 64, 3)))
    # model.add(BatchNormalization(axis=1, input_shape=(20,64,3)))
    model.add(Convolution2D(16, 3, 3, border_mode='valid', subsample=(2,2), activation='relu'))
    model.add(Convolution2D(24, 3, 3, border_mode='valid', subsample=(1,1), activation='relu'))
    model.add(Convolution2D(36, 3, 3, border_mode='valid', activation='relu'))
    # model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(48, 2, 2, border_mode='valid', activation='relu'))
    model.add(Convolution2D(48, 2, 2, border_mode='valid', activation='relu'))
    model.add(Flatten())
    model.add(Dense(512))
    # model.add(Activation('relu'))
    model.add(Dropout(.2))
    model.add(Activation('relu'))
    # model.add(Dense(50))
    # model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.summary()

    # model = Sequential()
    # model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(100, 200, 3)))
    # # model.add(BatchNormalization(axis=1, input_shape=(100, 200, 3)))
    # model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2,2), activation='relu'))
    # model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2,2), activation='relu'))
    # model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2,2), activation='relu'))
    # model.add(SpatialDropout2D(0.5))
    # model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
    # model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
    # model.add(Flatten())
    # # model.add(Activation('relu'))
    # model.add(Dense(100))
    # # model.add(Dropout(.25))
    # model.add(Activation('relu'))
    # model.add(Dense(50))
    # model.add(Activation('relu'))
    # model.add(Dense(10))
    # model.add(Activation('relu'))
    # model.add(Dense(num_classes))

    # model = Sequential()
    # # model.add(BatchNormalization(axis=1, input_shape=(32,16,3)))
    # model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(32, 16, 3)))
    # model.add(Convolution2D(16, 3, 3, border_mode='valid', subsample=(2,2), activation='relu'))
    # model.add(Convolution2D(24, 3, 3, border_mode='valid', subsample=(1,1), activation='relu'))
    # model.add(Convolution2D(36, 3, 3, border_mode='valid', activation='relu'))
    # model.add(Convolution2D(48, 2, 2, border_mode='valid', activation='relu'))
    # model.add(Convolution2D(48, 2, 2, border_mode='valid', activation='relu'))
    # model.add(Flatten())
    # # model.add(Dense(1164))
    # # model.add(Activation('relu'))
    # model.add(Dense(512))
    # model.add(Dropout(.25))
    # model.add(Activation('relu'))
    # model.add(Dense(10))
    # model.add(Activation('relu'))
    # model.add(Dense(1))
    # model.summary()

# model.add(BatchNormalization(axis=1, input_shape=(100, 200, 3)))
# model.add(Convolution2D(24, 3, 3, border_mode='valid', subsample=(2,2), activation='relu'))
# model.add(Convolution2D(36, 3, 3, border_mode='valid', subsample=(2,2), activation='relu'))
# model.add(Convolution2D(48, 3, 3, border_mode='valid', subsample=(2,2), activation='relu'))
# model.add(Convolution2D(64, 2, 2, border_mode='valid', activation='relu'))
# model.add(Convolution2D(64, 2, 2, border_mode='valid', activation='relu'))
# model.add(Flatten())
# model.add(Dense(512))
# model.add(Dropout(.5))
# model.add(Activation('relu'))
# model.add(Dense(10))
# model.add(Activation('relu'))
# model.add(Dense(num_classes))

# model.add(MaxPooling2D(pool_size=(2,3),input_shape=(100, 200, 3)))
# model.add(Lambda(lambda x: x/127.5 - 1.))
# model.add(Convolution2D(5, 5, 24, subsample=(4, 4), border_mode="same"))
# model.add(ELU())
# model.add(Convolution2D(5, 5, 36, subsample=(2, 2), border_mode="same"))
# model.add(ELU())
# model.add(Convolution2D(5, 5, 48, subsample=(2, 2), border_mode="same"))
# model.add(ELU())
# model.add(Convolution2D(3, 3, 64, subsample=(2, 2), border_mode="same"))
# model.add(ELU())
# model.add(Convolution2D(3, 3, 64, subsample=(2, 2), border_mode="same"))
# model.add(Flatten())
# model.add(Dropout(.2))
# model.add(ELU())
# model.add(Activation('relu'))
# model.add(Dense(1164))
# model.add(Dropout(.2))
# model.add(ELU())
# model.add(Activation('relu'))
# model.add(Dense(100))
# model.add(Dropout(.2))
# model.add(ELU())
# model.add(Activation('relu'))
# model.add(Dense(50))
# model.add(Dropout(.2))
# model.add(ELU())
# model.add(Activation('relu'))
# model.add(Dense(10))
# model.add(Dropout(.2))
# model.add(ELU())
# model.add(Activation('relu'))
# model.add(Dense(1))

# model.add(BatchNormalization(input_shape=(100, 200, 3)))
# model.add(Convolution2D(24, 5, 5, border_mode='same', activation='relu', W_constraint=maxnorm(3)))
# model.add(Convolution2D(36, 5, 5, border_mode='same', activation='relu', W_constraint=maxnorm(3)))
# model.add(Convolution2D(48, 5, 5, activation='relu', border_mode='same', subsample=(2, 2)))
# model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', subsample=(1, 1)))
# model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', subsample=(1, 1)))
# model.add(Dropout(0.2))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
# model.add(Dense(100, activation='relu', W_constraint=maxnorm(3)))
# model.add(Dense(50, activation='relu', W_constraint=maxnorm(3)))
# model.add(Dense(10, activation='relu', W_constraint=maxnorm(3)))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes))

    adam = Adam(lr=learning_rate, decay=0.001) #0.00001
    model.compile(loss='mean_squared_error', optimizer=adam)
    print(model.summary())

# Model will save the weights whenever validation loss improves
checkpoint = ModelCheckpoint(filepath = 'model.h5', verbose = 1, save_best_only=True, monitor='val_loss')

# Discontinue training when validation loss fails to decrease
callback = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

# #fit the model to the training data and use the validation set as validation data:
# model.fit(X_train, y_train, nb_epoch=1, batch_size=32, validation_data=(X_val, y_val))

# model.fit_generator(myGenerator(), samples_per_epoch=len(X_train), nb_epoch = 1,
#  validation_data=myValGenerator(), nb_val_samples =len(X_val), callbacks=[checkpoint, callback])

model.fit(X_train,
        y_train,
        nb_epoch=5,
        verbose=1,
        batch_size=128,
        shuffle=True,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, callback])



#saving the model and the weigths:
model_json = model.to_json()
with open('model.json', 'w') as f:
    json.dump(model_json, f)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")