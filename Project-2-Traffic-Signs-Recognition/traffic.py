import pickle

# TODO: fill this in based on where you saved the training and testing data
training_file = '/Users/michelecavaioni/Flatiron/My-Projects/Udacity (Self Driving Car)/Project #2 (Traffic Signs Recognition)/traffic-signs-data/train.p'
testing_file = '/Users/michelecavaioni/Flatiron/My-Projects/Udacity (Self Driving Car)/Project #2 (Traffic Signs Recognition)/traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']



n_train = len(X_train)

# TODO: number of testing examples
n_test = len(X_test)

# TODO: what's the shape of an image?
image_shape = X_train[0].shape

# TODO: how many classes are in the dataset
import numpy as np
n_classes = len(np.unique(y_train))

import cv2
def grayscale(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image_gray

X_train_gray=[]
for i in range(len(X_train)):
    X_train_gray.append(grayscale(X_train[i]))
X_train_gray = np.array(X_train_gray)
X_train_gray = X_train_gray[:,:,:,np.newaxis]

X_test_gray=[]
for i in range(len(X_test)):
    X_test_gray.append(grayscale(X_test[i]))
X_test_gray = np.array(X_test_gray)
X_test_gray = X_test_gray[:,:,:,np.newaxis]

#normalizing image data:
def normalize_greyscale(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    """
    a=0.1
    b=0.9
    x_min = 0
    x_max =255   
    return (a + (((image_data)*(b-a))/(x_max-x_min)))

X_train_norm = []
for i in range(len(X_train_gray)):
    X_train_norm.append(normalize_greyscale(X_train_gray[i]))
X_train_norm = np.array(X_train_norm)


X_test_norm = []
for i in range(len(X_test_gray)):
    X_test_norm.append(normalize_greyscale(X_test_gray[i]))    
X_test_norm = np.array(X_test_norm)

from sklearn.cross_validation import train_test_split

X_train_fin, X_val, y_train_fin, y_val = train_test_split(X_train_norm, y_train, test_size=0.33, random_state=42)

from keras.utils import np_utils

y_train_fin = np_utils.to_categorical(y_train_fin)
y_val = np_utils.to_categorical(y_val)

# import tensorflow as tf


# # Parameters
# learning_rate = 0.001
# training_epochs = 15
# batch_size = 100
# display_step = 1

# n_input = 1024  # Traffic sign data input (img shape: 32*32)
# n_classes = 43  # As calculated before this value is 43

# n_hidden_layer = 256 # layer number of features

# weights = {
#     'hidden_layer': tf.Variable(tf.random_normal([n_input, n_hidden_layer])),
#     'out': tf.Variable(tf.random_normal([n_hidden_layer, n_classes]))
# }
# biases = {
#     'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer])),
#     'out': tf.Variable(tf.random_normal([n_classes]))
# }

# x = tf.placeholder("float", [None, 32, 32, 1])
# y = tf.placeholder("float", [None, n_classes])

# x_flat = tf.reshape(x, [-1, n_input]) #reshapes a batch of 32*32 pixels, x, to a batch of 1024 pixels.

# # Hidden layer with RELU activation
# layer_1 = tf.add(tf.matmul(x_flat, weights['hidden_layer']), biases['hidden_layer'])
# layer_1 = tf.nn.relu(layer_1)
# # Output layer with linear activation
# logits = tf.matmul(layer_1, weights['out']) + biases['out']

# # Define loss and optimizer
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# # Initializing the variables
# init = tf.initialize_all_variables()

# # Launch the graph
# with tf.Session() as sess:
#     sess.run(init)
#     # Training cycle
#     for epoch in range(training_epochs):
#         total_batch = int(n_train/batch_size)
#         # Loop over all batches
#         batch_in =0
#         batch_fin = batch_size
#         for i in range(total_batch):
#             batch_x = X_train_fin[batch_in:batch_fin]
#             batch_y = y_train_fin[batch_in:batch_fin]
#             if (batch_fin+100) < len(y_train_fin):
#                 batch_fin = batch_fin + 100
#                 batch_in = batch_in + 100
#             else:
#                 batch_in = batch_in + 100
#                 batch_fin = len(y_train_fin)
#             # Run optimization op (backprop) and cost op (to get loss value)
#             sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

# correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
# print(correct_prediction)


import tensorflow as tf

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

n_input = 1024  # MNIST data input (img shape: 28*28)
n_classes = 43  # MNIST total classes (0-9 digits)

n_hidden_layer = 256 # layer number of features

# Store layers weight & bias
weights = {
    'hidden_layer': tf.Variable(tf.random_normal([n_input, n_hidden_layer])),
    'out': tf.Variable(tf.random_normal([n_hidden_layer, n_classes]))
}
biases = {
    'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# tf Graph input
x = tf.placeholder("float", [None, 32, 32, 1])
y = tf.placeholder("float", [None, n_classes])

x_flat = tf.reshape(x, [-1, n_input])

# Hidden layer with RELU activation
layer_1 = tf.add(tf.matmul(x_flat, weights['hidden_layer']), biases['hidden_layer'])
layer_1 = tf.nn.relu(layer_1)
# Output layer with linear activation
logits = tf.matmul(layer_1, weights['out']) + biases['out']

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        total_batch = int(n_train/batch_size)
        # Loop over all batches
        batch_in =0
        batch_fin = batch_size
        for i in range(total_batch):
            batch_x = X_train_fin[batch_in:batch_fin]
            batch_y = y_train_fin[batch_in:batch_fin]
            while (batch_fin+100) < len(y_train_fin):
              if (batch_fin+100) < len(y_train_fin):
                  batch_fin = batch_fin + 100
                  batch_in = batch_in + 100
              else:
                  batch_in = batch_in + 100
                  batch_fin = len(y_train_fin)
            # Run optimization op (backprop) and cost op (to get loss value)
              sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        # Display logs per epoch step
        if epoch % display_step == 0:
            c = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(c))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("Accuracy:", accuracy.eval({x: X_train_fin, y: y_train_fin}))
