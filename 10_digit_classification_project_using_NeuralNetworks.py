



#  10 digit classification project using Convolutional Neural Networks

# used code at this location for reference:
##  https://github.com/aymericdamien/TensorFlow-Examples

# learned how it works, did fine tuning, ...  by Ramin Monajemy


## 12A  import MNIST data for conv code trial

from __future__ import print_function

import tensorflow as tf

print('IMPORT START 1')

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

print('IMPORT DONE 1')

print('***  download done  ****')

print('MNIST DATA train images: ', mnist.train.images.shape)
print('MNIST DATA train labels: ', mnist.train.labels.shape)
print('MNIST DATA validation images: ', mnist.validation.images.shape)
print('MNIST DATA validation labels: ', mnist.validation.labels.shape)
print('MNIST DATA test images: ', mnist.test.images.shape)
print('MNIST DATA test labels: ', mnist.test.labels.shape)



# In[108]:


##12B VERIFIED (9/12/16) Convoulional network example for 10-class mnist digits
##  https://github.com/aymericdamien/TensorFlow-Examples


##########################
##  SEE https://www.tensorflow.org/versions/0.6.0/tutorials/mnist/pros/index.html 
## for details of how convoultions work in Tensorflow
###########################

'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf


# Parameters
#learning_rate = 0.001  # ORIGINAL STATEMENT
Learning_rate = 0.01

#training_iters = 200000
#training_iters = 5500
training_iters = 2500

batch_size = 128
display_step = 10

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')  # SAME means pad with zeros => output same as input size
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2): # for X = 8 by 8 block of data,  generate a 2x2 where each element is the max of its 2x2 counterpart in X
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])    # WC1  = 5, 5, 1, 32 (NOTE:  x,x,1,32 means single layer as input
                                                        #                            + 32 layers of the first hidden covnet)
                                                        # bc1  = 32  

    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    #Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])  # WC2 = 5, 5, 32, 64  (NOTE:  x,x,32,64 means 32 layers as inputs from above
                                                          #                            + 64 layers of the second hidden covnet)
                                                          # BC2 = 64 
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)  # 

    # CONV2 :  looks like a 7x7,  64 layers deep
    
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])  # reshaper to a (1)x(7x7x64) data vector    
    
    print("TEST",[-1, weights['wd1'].get_shape().as_list()[0]])
    
    # CONV2 :  looks like a 7x7,  64 layers deep
    
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])  # WD1 = (1, 1024) data vector
                                                                 # BD1 = 1024

    fc1 = tf.nn.relu(fc1)               #WORKING BACKWARDS:     fc1 = 1x1024
    # Apply Dropout
 
    fc1 = tf.nn.dropout(fc1, dropout)   # do drop out with probability of 0.75
                                        # With probability keep_prob, outputs the input element scaled up by 1 / keep_prob, 
                                        # otherwise outputs 0. The scaling is so that the expected sum is unchanged.

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])  # out is 1024x1 data vector X W_out = 1x10
    
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=Learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)   # batch size is 128
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " +                   "{:.6f}".format(loss) + ", Training Accuracy= " +                   "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:",         sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256],
                                      keep_prob: 1.}))
