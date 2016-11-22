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


# X_train_norm = []
# for i in range(len(X_train_gray)):
#     X_train_norm.append(normalize_greyscale(X_train_gray[i]))
# X_train_norm = np.array(X_train_norm)
X_train_norm = normalize_greyscale(X_train_gray)

# X_test_norm = []
# for i in range(len(X_test_gray)):
#     X_test_norm.append(normalize_greyscale(X_test_gray[i]))    
# X_test_norm = np.array(X_test_norm)
X_test_norm = normalize_greyscale(X_test_gray)


from sklearn.cross_validation import train_test_split

X_train_fin, X_val, y_train_fin, y_val = train_test_split(X_train_norm, y_train, test_size=0.33, random_state=42)

from keras.utils import np_utils

y_train_fin = np_utils.to_categorical(y_train_fin)
y_val = np_utils.to_categorical(y_val)



y_train_fin = y_train_fin.astype(np.float32)
y_val = y_val.astype(np.float32)
y_test = y_test.astype(np.float32)

X_train_fin = X_train_fin.astype(np.float32)

import tensorflow as tf



# Parameters
# learning_rate = 0.2
# training_epochs = 4
# batch_size = 100
# display_step = 1

n_input = 1024  # MNIST data input (img shape: 32*32)
n_classes = 43  # MNIST total classes (0-9 digits)
n_train_fin = len(X_train_fin)
n_labels_fin = len(y_train_fin)
n_hidden_layer = 256 # layer number of features

X_train_fin = X_train_fin.reshape([-1, n_input])
X_val = X_val.reshape([-1, n_input])

x = tf.placeholder(tf.float32)
y = tf.placeholder("float", [None, n_classes])

weights = tf.Variable(tf.truncated_normal((n_input, n_classes)))
biases = tf.Variable(tf.zeros(n_classes))

y_conv = tf.add(tf.matmul(x, weights), biases)

prediction = tf.nn.softmax(y_conv)

# Cross entropy
cross_entropy = -tf.reduce_sum(y * tf.log(prediction), reduction_indices=1)
# cross_entropy = -tf.reduce_sum(y * tf.log(tf.clip_by_value(prediction,1e-10,1.0)), reduction_indices=1)
# Training loss
loss = tf.reduce_mean(cross_entropy)

# Create an operation that initializes all variables
init = tf.initialize_all_variables()


train_feed_dict = {x: X_train_fin, y: y_train_fin}
valid_feed_dict = {x: X_val, y: y_val}
test_feed_dict = {x: X_test, y: y_test}
# Test Cases
with tf.Session() as session:
    session.run(init)
    # session.run(loss, feed_dict={x: (X_train_fin), y: (y_train_fin)})
    session.run(loss, feed_dict=train_feed_dict)
    session.run(loss, feed_dict=valid_feed_dict)
    # session.run(loss, feed_dict=test_feed_dict)
    # biases_data = session.run(biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
# # optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)
# Initializing the variables
# init = tf.initialize_all_variables()

# # Launch the graph
# with tf.Session() as sess:
#     sess.run(init)
#     # Training cycle
#     for epoch in range(training_epochs):
#         total_batch = int(n_train_fin/batch_size)
#         # Loop over all batches
#         batch_in =0
#         batch_fin = batch_size
#         for i in range(total_batch):
#             batch_x = X_train_fin[batch_in:batch_fin]
#             batch_y = y_train_fin[batch_in:batch_fin]
#             while (batch_fin+100) < len(y_train_fin):
#               if (batch_fin+100) < len(y_train_fin):
#                   batch_fin = batch_fin + 100
#                   batch_in = batch_in + 100
#               else:
#                   batch_in = batch_in + 100
#                   batch_fin = len(y_train_fin)
#             # Run optimization op (backprop) and cost op (to get loss value)
#               sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
#         # Display logs per epoch step
#         if epoch % display_step == 0:
#             c = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
#             print("Epoch:", '%04d' % (epoch+1), "cost=", \
#                 "{:.9f}".format(c))
#     print("Optimization Finished!")

#     # Test model on validaation set:
#     correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
#     # Calculate accuracy
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#     print("Validation Accuracy:", accuracy.eval({x: X_val, y: y_val}))

##########

is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
# Calculate the accuracy of the predictions
accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

import math
from tqdm import tqdm

epochs = 100
batch_size = 100
learning_rate = 0.11

### DON'T MODIFY ANYTHING BELOW ###
# Gradient Descent
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)    

# The accuracy measured against the validation set
validation_accuracy = 0.0

# Measurements use for graphing loss and accuracy
log_batch_step = 50
batches = []
loss_batch = []
train_acc_batch = []
valid_acc_batch = []

with tf.Session() as session:
    session.run(init)
    batch_count = int(math.ceil(len(X_train_fin)/batch_size))

    for epoch_i in range(epochs):
        
        # Progress bar
        batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i+1, epochs), unit='batches')
        
        # The training cycle
        for batch_i in batches_pbar:
            # Get a batch of training features and labels
            batch_start = batch_i*batch_size
            batch_features = X_train_fin[batch_start:batch_start + batch_size]
            batch_labels = y_train_fin[batch_start:batch_start + batch_size]

            # Run optimizer and get loss
            _, l = session.run(
                [optimizer, loss],
                feed_dict={x: batch_features, y: batch_labels})

            # Log every 50 batches
            if not batch_i % log_batch_step:
                # Calculate Training and Validation accuracy
                training_accuracy = session.run(accuracy, feed_dict=train_feed_dict)
                validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

                # Log batches
                previous_batch = batches[-1] if batches else 0
                batches.append(log_batch_step + previous_batch)
                loss_batch.append(l)
                train_acc_batch.append(training_accuracy)
                valid_acc_batch.append(validation_accuracy)

        # Check accuracy against Validation data
        validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)
print('Validation accuracy at {}'.format(validation_accuracy))
