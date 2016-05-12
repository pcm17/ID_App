# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tempfile

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable = redefined-builtin
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

# Automatically download and import the MNIST dataset
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Create a directory 'MNIST_data' in which to store the data files
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Begin the Session
sess = tf.InteractiveSession()

# Initialze x: The placeholder for input images. It will consist of a 2d tensor of floats
x = tf.placeholder(tf.float32, [None, 784])

# Initialize y: The placeholder for output classes. It will consist of a 2d tensor of floats
# Each row is a one-hot 10d vector indicating which digit class the corresponding MNIST image belongs to
y_ = tf.placeholder(tf.float32, [None, 10])

# Weight: A tensor initially full of zeros and lives in the graph
W = tf.Variable(tf.zeros([784, 10]))

# Bias: A tensor full of zeros that lives in the graph
b = tf.Variable(tf.zeros([10]))


# A function to create weights & initialize them with a small amount of noise for symmetry breaking/prevent 0 gradients
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# A function to create biases. We initialize them with a slightly positive initial bias to avoid "dead neurons."
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# ---------------------------------------------------------------------------------------------------------

# First Convolutional Layer: Consists of first convolution, followed by max pooling
# The convolutional will compute 32 features for each 5x5 patch
# The first 2 dimensions are the patch size (5 x 5)
# Next is the num of input channels (1)
# Last is the num of output channels (32)
W_conv1 = weight_variable([5, 5, 1, 32])

# b_conv1: A bias vector with a component for each output channel (32)
b_conv1 = bias_variable([32])

# Apply the first layer by reshaping x to a 4 dimensional tensor
# The 2nd and 3rd dimensions are the image width and height (28 x 28)
# The final dimension is the number of color channels (1)
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Convolve x_image with the W_conv1 and add the bias [conv2d(x_image, W_conv1) + b_conv1]
# Then apply the ReLU function (tf.nn.relu) and max pool the whole thing
h_pool1 = max_pool_2x2(tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1))

# ---------------------------------------------------------------------------------------------------------

# Second Convolutional Layer: Stack layers to build a deep network
# The second layer will compute 64 features for each 5x5 patch
# The first 2 dimensions are the patch size (5 x 5)
# Next is the num of input channels (32)
# Last is the num of output channels (64)
W_conv2 = weight_variable([5, 5, 32, 64])

# b_conv2: A bias vector with a component for each output channel (64)
b_conv2 = bias_variable([64])

# Convolve h_pool1 with the W_conv2 and add the bias [conv2d(h_pool1, W_conv2) + b_conv2]
# Then apply the ReLU function (tf.nn.relu) and max pool the whole thing
h_pool2 = max_pool_2x2(tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2))

# ---------------------------------------------------------------------------------------------------------

# Densely Connected Layer: Add a Fully-Connected layer with 1024 neurons to process the entire image
# The first 2 dimensions are the patch size (7 x 7)
# Next is the num of input channels (64)
# Last is the num of output channels (1024)
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

# Reshape the tensor from a pooling layer into a batch of vectors
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

# Multiply by a weight matrix (W_fc1), add a bias (b_fc1), and apply a ReLU
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# To reduce overfitting, we will apply dropout before the Readout layer
# Create a placeholder for the probability that a neuron's output is kept during dropout
# This allows us to turn dropout on during training, and turn it off during testing
keep_prob = tf.placeholder(tf.float32)

# tf.nn.dropout auto handles scaling neuron outputs as well as masking them, so it just works without additional scaling
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# ---------------------------------------------------------------------------------------------------------

# Readout Layer: The softmax layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
# Multiply by a weight matrix (W_fc2), add a bias (b_fc2), and apply a softmax
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# ---------------------------------------------------------------------------------------------------------

# Train and Evaluate the Model
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 10 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# ---------------------------------------------------------------------------------------------------------

# Print Final Test Accuracy
print("Test Accuracy: %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
