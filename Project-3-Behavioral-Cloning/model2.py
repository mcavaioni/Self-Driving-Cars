import os
from pathlib import Path
import numpy as np
from numpy.random import random
import cv2
import pandas as pd
import json
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, ELU
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import Adam

from sklearn.cross_validation import train_test_split
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


ch, img_rows, img_cols = 3, 66, 200

import pandas as pd
csv_file = '/Users/michelecavaioni/Flatiron/My-Projects/Udacity (Self Driving Car)/Project #3 (Behavioral Cloning)/driving_log_edit.csv'
data = pd.read_csv(csv_file)

X_train = np.copy(data['center']+':'+data['left']+':'+data['right'])
Y_train = np.copy(data['Steering Angle'])
Y_train = Y_train.astype(np.float32)

## split the training data into training and validation
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.3, random_state=10)



def load_image(imagepath):
    imagepath = imagepath.replace(' ','')
    image = cv2.imread(imagepath, 1)
    # get shape and chop off 1/3 from the top
    shape = image.shape
    # note: numpy arrays are (row, col)!
    image = image[int(shape[0]/3):shape[0], 0:shape[1]]
    image = cv2.resize(image, (img_cols, img_rows))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image

# our image generator.
def batchgen(X, Y):
    while 1:
        for i in range(len(X)):
            y = Y[i]
            if y < -0.01:
                chance = random()
                if chance > 0.75:
                    imagepath = X[i].split(':')[1]
                    y *= 3.0
                else:
                    if chance > 0.5:
                        imagepath = X[i].split(':')[1]
                        y *= 2.0
                    else:
                        if chance > 0.25:
                           imagepath = X[i].split(':')[0]
                           y *= 1.5
                        else:
                           imagepath = X[i].split(':')[0]
            else:
                if y > 0.01:
                    chance = random()
                    if chance > 0.75:
                        imagepath = X[i].split(':')[2]
                        y *= 3.0
                    else:
                        if chance > 0.5:
                            imagepath = X[i].split(':')[2]
                            y *= 2.0
                        else:
                            if chance > 0.25:
                                imagepath = X[i].split(':')[0]
                                y *= 1.5
                            else:
                                imagepath = X[i].split(':')[0]
                else:
                    imagepath = X[i].split(':')[0]
            image = load_image(imagepath)
            y = np.array([[y]])
            image = image.reshape(1, img_rows, img_cols, ch)
            yield image, y

fine_tune_mode = False

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
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(img_rows, img_cols, ch)))
    # model.add(BatchNormalization(axis=1, input_shape=(20,64,3)))
    model.add(Convolution2D(24, 3, 3, border_mode='valid', subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36, 3, 3, border_mode='valid', subsample=(1,1), activation='relu'))
    model.add(Convolution2D(48, 3, 3, border_mode='valid', activation='relu'))
    # model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(64, 2, 2, border_mode='valid', activation='relu'))
    model.add(Convolution2D(64, 2, 2, border_mode='valid', activation='relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(.2))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.summary()  

    adam = Adam(lr=learning_rate, decay=0.001) #0.00001
    model.compile(loss='mean_squared_error', optimizer=adam)
    print(model.summary())

checkpoint = ModelCheckpoint(filepath = 'model.h5', verbose = 1, save_best_only=True, monitor='val_loss')

# Discontinue training when validation loss fails to decrease
callback = EarlyStopping(monitor='val_loss', patience=2, verbose=1)


model.fit_generator(batchgen(X_train, Y_train), samples_per_epoch=len(X_train)/20, nb_epoch = 1,
 validation_data=batchgen(X_val, Y_val), nb_val_samples =len(X_val), callbacks=[checkpoint, callback])  

# history = model.fit_generator(batchgen(X_train, Y_train),
#                     samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch,
#                     validation_data=batchgen(X_val, Y_val),
#                     nb_val_samples=val_size,
#                     verbose=1)

model_json = model.to_json()
with open('model.json', 'w') as f:
    json.dump(model_json, f)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
