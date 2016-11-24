import pickle
from keras.utils import np_utils
# TODO: fill this in based on where you saved the training and testing data
training_file = '/Users/michelecavaioni/Flatiron/My-Projects/Udacity (Self Driving Car)/Project #2 (Traffic Signs Recognition)/traffic-signs-data/train.p'
testing_file = '/Users/michelecavaioni/Flatiron/My-Projects/Udacity (Self Driving Car)/Project #2 (Traffic Signs Recognition)/traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

######new test sample of 5 images
# import glob, os
# import cv2
# import numpy as np

# path = '/Users/michelecavaioni/Flatiron/My-Projects/Udacity (Self Driving Car)/Project #2 (Traffic Signs Recognition)/traffic_sign_images/32x32pix/'
# image_dict = []

# for i,infile in enumerate(glob.glob(os.path.join(path,'*.png'))):
#     img = cv2.imread(infile)
#     # my_key = i              
#     image_dict.append(img) 
# image_dict=image_dict[0:5]
# X_test = np.array(image_dict)

# y_test = np.array([30,17,25, 13, 31])
y_test = np_utils.to_categorical(y_test)
# new_y = []
# for i in range(len(y_test)):
#   single = np.append(y_test[i], [0,0,0,0,0,0,0,0,0,0,0])
#   new_y.append(single)
# y_test = np.array(new_y)
############


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

X_train_fin, X_val, y_train_fin, y_val = train_test_split(X_train_norm, y_train, test_size=0.05, random_state=42)

from keras.utils import np_utils

y_train_fin = np_utils.to_categorical(y_train_fin)
y_val = np_utils.to_categorical(y_val)
# y_test = np_utils.to_categorical(y_test)



import tensorflow as tf
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

y_train_fin = y_train_fin.astype(np.float32)
y_val = y_val.astype(np.float32)
y_test = y_test.astype(np.float32)

X_train_fin = X_train_fin.astype(np.float32)
X_test_norm = X_test_norm.astype(np.float32)



# Parameters
epochs = 1
batch_size = 50
learning_rate = 0.01

n_input = 1024  # data input (img shape: 32*32)
n_classes = 43  # total classes 
n_train_fin = len(X_train_fin)
n_labels_fin = len(y_train_fin)

X_train_fin = X_train_fin.reshape([-1, n_input])
X_val = X_val.reshape([-1, n_input])
X_test_norm = X_test_norm.reshape([-1, n_input])


x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)


weights = tf.Variable(tf.truncated_normal((n_input, n_classes)))
biases = tf.Variable(tf.zeros(n_classes))

y_conv = tf.add(tf.matmul(x, weights), biases)

prediction = tf.nn.softmax(y_conv)

# Cross entropy
cross_entropy = -tf.reduce_sum(y * tf.log(prediction), reduction_indices=1)
# Training loss
cost = tf.reduce_mean(cross_entropy)

# Create an operation that initializes all variables
init = tf.initialize_all_variables()


train_feed_dict = {x: X_train_fin, y: y_train_fin}
valid_feed_dict = {x: X_val, y: y_val}
test_feed_dict = {x: X_test_norm, y: y_test}

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y))

# Train and Validation Cases
with tf.Session() as session:
    session.run(init)
    session.run(cost, feed_dict=train_feed_dict)
    session.run(cost, feed_dict=valid_feed_dict)
    session.run(cost, feed_dict=test_feed_dict)
    biases_data = session.run(biases)


is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
# Calculate the accuracy of the predictions
accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

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
                [optimizer, cost],
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

loss_plot = plt.subplot(211)
loss_plot.set_title('Loss')
loss_plot.plot(batches, loss_batch, 'g')
loss_plot.set_xlim([batches[0], batches[-1]])
acc_plot = plt.subplot(212)
acc_plot.set_title('Accuracy')
acc_plot.plot(batches, train_acc_batch, 'r', label='Training Accuracy')
acc_plot.plot(batches, valid_acc_batch, 'b', label='Validation Accuracy')
acc_plot.set_ylim([0, 1.0])
acc_plot.set_xlim([batches[0], batches[-1]])
acc_plot.legend(loc=4)
plt.tight_layout()
plt.show()

# The accuracy measured against the Test set

test_accuracy = 0.0

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

            # Run optimizer
            _ = session.run(optimizer, feed_dict={x: batch_features, y: batch_labels})

        # Check accuracy against Test data
        test_accuracy = session.run(accuracy, feed_dict=test_feed_dict)


print('Done! Test Accuracy is {}'.format(test_accuracy))
