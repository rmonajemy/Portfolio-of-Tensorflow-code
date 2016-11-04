
# coding: utf-8

# Deep Learning
# =============
# 
# Assignment 4
# ------------
# 
# Previously in `2_fullyconnected.ipynb` and `3_regularization.ipynb`, we trained fully connected networks to classify [notMNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html) characters.
# 
# The goal of this assignment is make the neural network convolutional.

# In[ ]:

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range


# In[ ]:

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)


# Reformat into a TensorFlow-friendly shape:
# - convolutions need the image data formatted as a cube (width by height by #channels)
# - labels as float 1-hot encodings.

# In[ ]:

image_size = 28
num_labels = 10
num_channels = 1 # grayscale

import numpy as np

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


# In[ ]:

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


# Let's build a small network with two convolutional layers, followed by one fully connected layer. Convolutional networks are more expensive computationally, so we'll limit its depth and number of fully connected nodes.

# In[ ]:

batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64

graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  layer1_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, num_channels, depth], stddev=0.1))   ## dimentions:  5 x 5 x 1 x 16
  layer1_biases = tf.Variable(tf.zeros([depth]))                    ## dimentions:  1 x 16
  layer2_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth, depth], stddev=0.1))          ## dimentions:  5 x 5 x 16 x 16
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))      ## dimentions: don't know
  layer3_weights = tf.Variable(tf.truncated_normal(
      [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))   ## 28 //4 * 28 // 4 * 16:  whats the // operator?
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden])) ## what does tf.constant(1.0,64) do?
  layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_labels], stddev=0.1))                        ## dimentions:  64 x 10
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels])) ## dimentions:  what does constant(1.0, 64) do?
  
  # Model.
  def model(data):
    conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer1_biases)
    conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer2_biases)
    shape = hidden.get_shape().as_list()
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    return tf.matmul(hidden, layer4_weights) + layer4_biases
  
  # Training computation.
  logits = model(tf_train_dataset)
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  test_prediction = tf.nn.softmax(model(tf_test_dataset))


# In[1]:

tf.constant(1.0, 5)


# In[ ]:

num_steps = 1001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


# In[1]:

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


# In[27]:


#13A  import data for edge detection

from __future__ import print_function

import tensorflow as tf

print('IMPORT START 1')

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

print('IMPORT DONE 1')


print('MNIST DATA train images: ', mnist.train.images.shape)
print('MNIST DATA train labels: ', mnist.train.labels.shape)
print('MNIST DATA validation images: ', mnist.validation.images.shape)
print('MNIST DATA validation labels: ', mnist.validation.labels.shape)
print('MNIST DATA test images: ', mnist.test.images.shape)
print('MNIST DATA test labels: ', mnist.test.labels.shape)


# In[28]:




#13B:  my first project:  run a [-1,1] kernel on a picture to detect edges
#         -1   horizontal and veritical
#         -2   sqrt(x^2 + y^2) which should return the a shape similar to original


#print a sample first

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

N = 1001 #   thick O
x1_test = mnist.train.images[N][:]


# next lines just for plotting purposes
x1 = x1_test.reshape(28, 28)
_, (ax1) = plt.subplots(1, 1)
#ax1.imshow(x1_test.reshape(28, 28), cmap=plt.cm.Greys);
ax1.imshow(x1, cmap=plt.cm.Greys)
ax1.set_ylabel("y: 28 point")
ax1.set_xlabel("x: 28 point")
ax1.set_title("PLOT Nth CHARACTER")


# now build the TensoFlow algorythm





# ---
# Problem 1
# ---------
# 
# The convolutional model above uses convolutions with stride 2 to reduce the dimensionality. Replace the strides by a max pooling operation (`nn.max_pool()`) of stride 2 and kernel size 2.
# 
# ---

# In[22]:


import tensorflow as tf
import numpy as np

N = 1001 #   thick O
x1_test = mnist.train.images[N:N+1][:]
x = x1_test
x = tf.reshape(x, shape=[-1, 28, 28, 1])
print("shape of x is:  " , x)


#1  define parameters and filter
FILTER = np.random.random_sample((1,2,1,1))
FILTER[0][0][0][0] = -1
FILTER[0][1][0][0] = 1

print("FILTER: ", FILTER)

KERNEL = tf.constant(FILTER, dtype=tf.float32)
print("KERNET TF: ", KERNEL)



strides = 1


#2   Construct model
x_out = tf.nn.conv2d(x, KERNEL, strides=[1, strides, strides, 1], padding='SAME')  # SAME means pad with zeros => output same as input size
out1 = tf.reshape(x_out, [-1,28,28,1])

#3  Initializing the variables
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    
   
    out = sess.run(x_out) #, feed_dict={x_K: K1})
    
   
    print("OUT = ", out.shape)
    #TEST = tf.reshape(x_out, [-1, weights['wd1'].get_shape().as_list()[0]]) 
   # print("TEST",[-1, weights['wd1'].get_shape().as_list()[0]])
    
    print("KERNEL = ", KERNEL.eval())
    
    # next lines just for plotting purposes
   

    #out4plot = out[0][2][:][0]
    print("OUT4PLOT = ", out4plot.shape)
    _, (ax1) = plt.subplots(1, 1)
    #ax1.imshow(x1_test.reshape(28, 28), cmap=plt.cm.Greys);
    ax1.imshow(out4plot)
    ax1.set_ylabel("y: 28 point")
    ax1.set_xlabel("x: 28 point")
    ax1.set_title("PLOT Nth CHARACTER")






# ---
# Problem 2
# ---------
# 
# Try to get the best performance you can using a convolutional net. Look for example at the classic [LeNet5](http://yann.lecun.com/exdb/lenet/) architecture, adding Dropout, and/or adding learning rate decay.
# 
# ---

# In[26]:



#14  VERIFIED  JUST RUN A 3x3 image and run a [-1,1] kernel using CONV2D

import tensorflow as tf
import numpy as np



x1_test = np.array([1,2,4,4,7,11,7,12,18])
x = x1_test

x = tf.convert_to_tensor(x,dtype=tf.float32)
x = tf.reshape(x, shape=[-1, 3, 3, 1])                #WORK ON DETAILS OF THIS
print("shape of x is:  " , x)


#1  define parameters and filter for proper input into conv2d
FILTER = np.random.random_sample((1,2,1,1))   # filter input into conv2d:  rows, columns, input layers, output layers
FILTER[0][0][0][0] = -1
FILTER[0][1][0][0] = 1

print("FILTER: ", FILTER)

KERNEL = tf.constant(FILTER, dtype=tf.float32)
print("KERNET TF: ", KERNEL)



strides = 1


#2   Construct model
x_out = tf.nn.conv2d(x, KERNEL, strides=[1, strides, strides, 1], padding='SAME')  # SAME means pad with zeros => output same as input size


#3  Initializing the variables
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    
   
    out = sess.run(x_out) #, feed_dict={x_K: K1})
    
   
    print("OUT1 = ", out)
    
    
    FIG = np.zeros([3,3])
    for i in range(0,3):
      for j in range(0,3):
        FIG[i][j] = out[0][i][j][0]
    print(FIG)    


# In[23]:

#16A  VERIFEID load data  

import tensorflow as tf

print('IMPORT START 1')

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

print('IMPORT DONE 1')


print('MNIST DATA train images: ', mnist.train.images.shape)
print('MNIST DATA train labels: ', mnist.train.labels.shape)
print('MNIST DATA validation images: ', mnist.validation.images.shape)
print('MNIST DATA validation labels: ', mnist.validation.labels.shape)
print('MNIST DATA test images: ', mnist.test.images.shape)
print('MNIST DATA test labels: ', mnist.test.labels.shape)


    


# In[45]:

#16B  VERIFIED run [-1,1] and [-1,1]' kernel using CONV2D on a 28x28 image and extract edges in 2D

# USE 16A to import data
import tensorflow as tf
import numpy as np


#print a sample first

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


from __future__ import print_function



CHARACTER_OR_IMAGE = 0

if (CHARACTER_OR_IMAGE == 1):

#A  use a character from MNIST data

    N = 1001 #   thick O
    x1_test = mnist.train.images[N][:]


    # NOW PREPARE THE TENSOR FLOW
    x = x1_test
    DATA = x.reshape(28,28)

    x = tf.convert_to_tensor(x,dtype=tf.float32)
    x = tf.reshape(x, shape=[-1, 28, 28, 1])                #WORK ON DETAILS OF THIS
    print("shape of x is:  " , x)
    M = 28
    N= 28

else:
    
    #B  use a parrot
    image = plt.imread("parrot.png")
    #image = plt.imread("ID-100137333.jpg")
    #image = plt.imread("bridge.jpg")

    
    M = image.shape[0]
    N = image.shape[1]
    FIG1 = np.zeros([M,N])

    for i in range(M):
          for j in range(N):
            FIG1[i][j] = image[i][j][0]

    DATA = FIG1
    x = tf.convert_to_tensor(FIG1,dtype=tf.float32)
    x = tf.reshape(x, shape=[-1, M, N, 1])                #WORK ON DETAILS OF THIS
    print("shape of x is:  " , x)


    

#1  define parameters and filter for proper input into conv2d

#KERNEL 1 [-1,1]
FILTER = np.random.random_sample((1,2,1,1))   # filter input into conv2d:  rows, columns, input layers, output layers
FILTER[0][0][0][0] = -1
FILTER[0][1][0][0] = 1

KERNEL1 = tf.constant(FILTER, dtype=tf.float32)

#KERNEL B [-1,1]'

FILTER = np.random.random_sample((2,1,1,1))   # filter input into conv2d:  rows, columns, input layers, output layers
FILTER[0][0][0][0] = -1
FILTER[1][0][0][0] = 1

KERNEL2 = tf.constant(FILTER, dtype=tf.float32)


strides = 1


#2   Construct model
x_out1 = tf.nn.conv2d(x, KERNEL1, strides=[1, strides, strides, 1], padding='SAME')  # SAME means pad with zeros => output same as input size
x_out2 = tf.nn.conv2d(x, KERNEL2, strides=[1, strides, strides, 1], padding='SAME')  # SAME means pad with zeros => output same as input size


#3  Initializing the variables
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    
    out1 = sess.run(x_out1) #, feed_dict={x_K: K1})
    out2 = sess.run(x_out2) #, feed_dict={x_K: K1})

    
    
    FIG1 = np.zeros([M,N])
    for i in range(0,M):
      for j in range(0,N):
        FIG1[i][j] = out1[0][i][j][0]
    
    FIG2 = np.zeros([M,N])
    for i in range(0,M):
      for j in range(0,N):
        FIG2[i][j] = out2[0][i][j][0]
    
    FIG = np.zeros([M,N])
    for i in range(0,M):
      for j in range(0,N):
        FIG[i][j] = FIG1[i][j]**2   + FIG2[i][j]**2
    
    print([x1_test.max(), x1_test.min()])
    print([FIG1.max(), FIG1.min()])
    print([FIG2.max(), FIG2.min()])
    print([FIG.max(), FIG.min()])
    
  

# just the original and end result
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image) #sees 0 white, 1 black
    ax2.imshow(FIG,  cmap = plt.cm.Greys)


# four subplots
    _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    
    
    
    
    ax1.imshow(DATA, cmap=plt.cm.Greys, vmin = 0, vmax = 1); # cmap=plt.cm.Greys sees min(x) white, max(x) black
    #ax1.set_ylabel("y: 28 point")
    #ax1.set_xlabel("x: 28 point")
    ax1.set_title("input figure")
    
    ax2.imshow(FIG1, cmap=plt.cm.Greys, vmin = -1, vmax = 1);
    ax2.set_title("kernel=[-1,1]")
    
    ax3.imshow(FIG2, cmap=plt.cm.Greys, vmin = -1, vmax = 1);
    ax3.set_title("kernel=[-1,1]'")
    
    ax4.imshow(FIG, cmap=plt.cm.Greys) #, vmin = 0, vmax = 2);
    ax4.set_title("k1^2 + K2^2")
    
    #QUESTIONS / NEXT SUBTASKS
    #1  range of numbers that generate white (255) to black (0)  if vmin = 0 and vmax = 255 
       # plt.imshow(X, cmap = plt.get_cmap('gray'), vmin = 0, vmax = 255)
    #2  why the two middle plots are gray in background and first/last white?
    #3  what does x = tf.reshape(x, shape=[-1, 28, 28, 1])  do (-1?)
       #  -1 means the size of that dimension is computed so that the total size remains constant. 
    
    #4  why did x^2 + y^2 work although it ignores signs?
    #5  study  x = tf.placeholder(tf.float32, [None, n_input]), constant and variable definitions
    #6   # tensor 't' is [[[1, 1], [2, 2]],
         #                [[3, 3], [4, 4]]]
         # tensor 't' has shape [2, 2, 2]   WHY
    
    
    #Updates
    #1  feed data usign feeddict
    #2  use just zeros to define the filters, not random numbers
    #3  use subroutines instead of above ,  in page #17  :  PUBLISH ON GIT
    #4  implement example of particle filters in TensorFlow
    #5  implement a gradient descent 2D in TensorFlow,  identify a line 
    #     -  also a linear separator
    #     -  do the eigenvalue/vector analysis on same data / it should yield same results
    
    
    
    


# In[43]:

imgplot1 = plt.imshow(DATA, cmap=plt.cm.Greys)

imgplot2 = plt.imshow(FIG, cmap=plt.cm.Greys)

_, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(DATA) #sees 0 white, 1 black
ax2.imshow(FIG,  cmap = plt.cm.Greys)


x1_test = np.array([0,0,0,0,  0,0,0,0, 1,1,1,1,  1,1,1,1])
_, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(x1_test.reshape(4,4), cmap = plt.cm.Greys,         vmin = 0, vmax = 1) #sees 0 white, 1 black
ax2.imshow(x1_test.reshape(4,4), cmap = plt.get_cmap('gray'), vmin = 0, vmax = 1) #sees 0 black, 1 white



# In[33]:

#17B  try different kernels
#   combined  edge detecotr (two in  one)  [1,-1],[-1,1]


# USE 16A to import data
import tensorflow as tf
import numpy as np


#print a sample first

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


from __future__ import print_function



N = 1001 #   thick O
x1_test = mnist.train.images[N][:]




# NOW PREPARE THE TENSOR FLOW
x = x1_test

x = tf.convert_to_tensor(x,dtype=tf.float32)
x = tf.reshape(x, shape=[-1, 28, 28, 1])                #WORK ON DETAILS OF THIS
print("shape of x is:  " , x)


#1  define parameters and filter for proper input into conv2d

#KERNEL 1  edge detect
FILTER = np.random.random_sample((5,5,1,1))   # filter input into conv2d:  rows, columns, input layers, output layers

KK = np.array(    [[0,  0,  0,  0, 0], 
                   [0,  0,  1,  0, 0],
                   [0,  1, -5,  1, 0],
                   [0,  0,  1,  0, 0],
                   [0,  0,  0,  0, 0]])
for i in range(len(KK)):
    for j in range(len(KK)):
        FILTER[i][j][0][0] = KK[i][j]
        
KERNEL1 = tf.constant(FILTER, dtype=tf.float32)


#KERNEL B  sharpen

FILTER = np.random.random_sample((5,5,1,1))   # filter input into conv2d:  rows, columns, input layers, output layers
KK = np.array(    [[0,  0,  0,  0, 0], 
                   [0,  0, -1,  0, 0],
                   [0, -1,  5, -1, 0],
                   [0,  0, -1,  0, 0],
                   [0,  0,  0,  0, 0]])
for i in range(len(KK)):
    for j in range(len(KK)):
        FILTER[i][j][0][0] = KK[i][j]

KERNEL2 = tf.constant(FILTER, dtype=tf.float32)


strides = 1


#2   Construct model
x_out1 = tf.nn.conv2d(x, KERNEL1, strides=[1, strides, strides, 1], padding='SAME')  # SAME means pad with zeros => output same as input size
x_out2 = tf.nn.conv2d(x, KERNEL2, strides=[1, strides, strides, 1], padding='SAME')  # SAME means pad with zeros => output same as input size


#3  Initializing the variables
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    
    out1 = sess.run(x_out1) #, feed_dict={x_K: K1})
    out2 = sess.run(x_out2) #, feed_dict={x_K: K1})

    
    
    FIG1 = np.zeros([28,28])
    for i in range(0,28):
      for j in range(0,28):
        FIG1[i][j] = out1[0][i][j][0]
        FIG2[i][j] = out2[0][i][j][0]
        
        
    _, (ax1, ax2, ax3) = plt.subplots(1, 3)
    

    ax1.imshow(x1_test.reshape(28, 28), cmap=plt.cm.Greys, vmin = 0, vmax = 1); # cmap=plt.cm.Greys sees min(x) white, max(x) black
    #ax1.set_ylabel("y: 28 point")
    #ax1.set_xlabel("x: 28 point")
    ax1.set_title("input figure")
    
    ax2.imshow(FIG1, cmap=plt.cm.Greys);
    ax2.set_title("kernel=FILTER1")
    
    ax3.imshow(FIG2, cmap=plt.cm.Greys);
    ax3.set_title("kernel=FILTER2")
    
    
    
    #QUESTIONS / NEXT SUBTASKS
    #1  range of numbers that generate white (255) to black (0)  if vmin = 0 and vmax = 255 
       # plt.imshow(X, cmap = plt.get_cmap('gray'), vmin = 0, vmax = 255)
    #2  why the two middle plots are gray in background and first/last white?
    #3  what does x = tf.reshape(x, shape=[-1, 28, 28, 1])  do (-1?)
       #  -1 means the size of that dimension is computed so that the total size remains constant. 
    
    #4  why did x^2 + y^2 work although it ignores signs?
    #5  study  x = tf.placeholder(tf.float32, [None, n_input]), constant and variable definitions
    #6   # tensor 't' is [[[1, 1], [2, 2]],
         #                [[3, 3], [4, 4]]]
         # tensor 't' has shape [2, 2, 2]   WHY
    
    
    #Updates
    #1  feed data usign feeddict
    #2  use just zeros to define the filters, not random numbers
    #3  use subroutines instead of above ,  in page #17  :  PUBLISH ON GIT
    #4  implement example of particle filters in TensorFlow
    #5  implement a gradient descent 2D in TensorFlow,  identify a line 
    #     -  also a linear separator
    #     -  do the eigenvalue/vector analysis on same data / it should yield same results
    
    
    
    


# In[3]:

import numpy as np
FILTER = np.array([[0,  0,  0,  0, 0], 
                   [0,  0, -1,  0, 0],
                   [0, -1,  5, -1, 0],
                   [0,  0, -1,  0, 0],
                   [0,  0,  0,  0, 0]])
print(FILTER)

FILTER[2][2] = 55
print(FILTER)


# In[47]:


# PROJECT import picutre, process and analyze it

# FREE PICTURES
#   http://www.freedigitalphotos.net/images/Birds_g52-Red_Macaw_p137333.html


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
#from imread import imread, imsave
get_ipython().magic(u'matplotlib inline')
"""
def plot_image(label, image):
    #print label, image.shape
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.show()
"""
#IMG_PATH = "C:\Users\Ramin.Shohreh-VAIO\Documents\Ramin\STUDIES\AI DEEP LEARNING\DeepLearning\pictures\ID-100137333_parrot.jpeg"
#IMG_PATH = "c:\ID-100137333.jpg"
#img=mpimg.imread(IMG_PATH)



#img=mpimg.imread(parrot.png)

image = plt.imread("parrot.png")
#image = mping.imread(IMG_PATH)

imgplot = plt.imshow(image)
#plot_image("Original", image)


# In[31]:




# In[92]:

#[]


# In[31]:

# PROJECT import picutre, process and analyze it

# FREE PICTURES
#   http://www.freedigitalphotos.net/images/Birds_g52-Red_Macaw_p137333.html

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

get_ipython().magic(u'matplotlib inline')

#image = plt.imread("parrot.png")
#image = plt.imread("ID-100137333.jpg")
image = plt.imread("bridge.jpg")

imgplot = plt.imshow(image)
print(image.shape)
M = image.shape[0]
N = image.shape[1]
FIG1 = np.zeros([M,N])
FIG2 = np.zeros([M,N])
FIG3 = np.zeros([M,N])
FIG4 = np.zeros([M,N])
for i in range(M):
      for j in range(N):
        FIG1[i][j] = image[i][j][0]
        FIG2[i][j] = image[i][j][1]
        FIG3[i][j] = image[i][j][2]
        #FIG4[i][j] = image[i][j][0]
      
print(FIG1.shape)    
_, (ax1, ax2, ax3, ax4, qx5) = plt.subplots(1, 5)
    

        
    
ax1.imshow(image); # cmap=plt.cm.Greys sees min(x) white, max(x) black               
ax2.imshow(FIG1)                     
ax3.imshow(FIG2)
ax4.imshow(FIG3); # cmap=plt.cm.Greys sees min(x) white, max(x) black               
#ax5.imshow(FIG1)                     


# In[ ]:




# In[50]:

#18 try different kernels  on a parrot, bridge or a a simple black and white matrix
#   
# FREE PICTURES
#   http://www.freedigitalphotos.net/images/Birds_g52-Red_Macaw_p137333.html

# USE 16A to import data
import tensorflow as tf
import numpy as np


#print a sample first

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


from __future__ import print_function

#image = plt.imread("bridge.jpg")  
image = plt.imread("parrot.png")    # parrt.png is 360x412x4
#image = plt.imread("ID-100137333.jpg")




#imgplot = plt.imshow(image)
print(image.shape)
M = image.shape[0]
N = image.shape[1]
P1 = np.zeros([M,N])
for i in range(M):
      for j in range(N):
        P1[i][j] = image[i][j][0]  # CHOOSE LEVEL 0 OF 4 LAYERS THAT DEFINE COLOR AND ATTRIBUTES OF IMAGE
           

# TEST MY EDGE DETECTOR USING A SIMPLE MATRIX            
MAT = np.array(  [ [0,  0,  0,  0, 0, 0,  0,  0,  0, 0, 0,  0,  0,  0, 0], 
                   [0,  0,  0,  0, 0, 0,  0,  0,  0, 0, 0,  0,  0,  0, 0], 
                   [0,  0,  0,  0, 0, 0,  0,  0,  0, 0, 0,  0,  0,  0, 0],             
                   [0,  0,  0,  100, 100, 100,  100,  100,  100, 100, 0,  0,  0,  0, 0], 
                   [0,  0,  0,  100, 100, 100,  100,  100,  100, 100, 0,  0,  0,  0, 0], 
                   [0,  0,  0,  100, 100, 100,  100,  100,  100, 100, 0,  0,  0,  0, 0], 
                   [0,  0,  0,  100, 100, 100,  100,  100,  100, 100, 0,  0,  0,  0, 0], 
                   [0,  0,  0,  100, 100, 100,  100,  100,  100, 100, 0,  0,  0,  0, 0],
                   [0,  0,  0,  0, 0, 0,  0,  0,  0, 0,  0,  0,  0,  0, 0], 
                   [0,  0,  0,  0, 0, 0,  0,  0,  0, 0,  0,  0,  0,  0, 0]])
            

    
    
    
# NOW PREPARE the variable for THE TENSOR FLOW
x = P1     # USE THIS TO PROCESS PARROT OR BRIDGE IMAGE


P1 = MAT   # USE MATRIX AS INPUT COMMENT OUT THESE FOUR LINES TO USE ONE OF THE IMAGES (PARROT OR BRIDGE) INSTEAD IF SIMPLE MATRIX
x = MAT
M = 10
N = 15



x = tf.convert_to_tensor(x,dtype=tf.float32)  # convert integer array to float32 tensor  (NOTE: (A) CONVERT TO TENSOR  (B) RECAST AS FLOAT32)
print('convert inout to float32 tensor: ', x)

#x = tf.reshape(x, shape=[-1, 1,M*N, 1])             #CONV needs a 4-dim (rank 4) input  DONT NEED THIS
#print(x)
x = tf.reshape(x, shape=[-1, M, N, 1])               # TENSORFLOW CONV COMMANDS NEED RANK OF TENSOR TO BE 4 (4 DIMENTIOANL TENSOR)
#  -1 is inferred such that total number of emelements does not change
print('convert tensor to rank 4 (4-dimentional tensor) for proper input to conv() command: ', x)

# THIS STATEMENT NEEDED FOR PROPER IMAGE PROCESSING FOR TENSOR FLOW FOR IMAGES,  BUT WHY? 
#x = tf.convert_to_tensor(x,dtype=tf.float64)              # could not convert using float32  REMOVE JUST FOR 10x10 example


#x = tf.cast(x, tf.float32)  # tensor flow code needs tensors in float32  (needed if origianl variable is FLOAT64 LIKE SOME IMAGES)





#1  define parameters and filter for proper input into conv2d

#######################################################################
#KERNEL 1  edge detect
FILTER = np.random.random_sample((5,5,1,1))   # filter input into conv2d:  rows, columns, input layers, output layers

KK = np.array(    [[0.,  0.,  0.,  0., 0.], 
                   [0.,  0.,  1.,  0., 0.],
                   [0.,  1., -4.,  1., 0.],
                   [0.,  0.,  1.,  0., 0.],
                   [0.,  0.,  0.,  0., 0.]])
KK = KK*5

for i in range(len(KK)):
    for j in range(len(KK)):
        FILTER[i][j][0][0] = KK[i][j]  # prepare the Kenel to meet TensorFlow convolusion command requirements  [data][data], [input layesr],[output la]
        
KERNEL1 = tf.constant(FILTER, dtype=tf.float32)

#######################################################################
#KERNEL 2  trial

FILTER = np.random.random_sample((5,5,1,1))   # SHARPER IMAGE filter input into conv2d:  rows, columns, input layers, output layers
KK = np.array(    [[0.,  -1.,  1.,  0., 0.], 
                   [0.,  -1.,  1.,  0., 0.],
                   [0.,  -1.,  1.,  0., 0.],
                   [0.,  -1.,  1.,  0., 0.],
                   [0.,  -1.,  1.,  0., 0.]])
               
KK = KK*10    
    
for i in range(len(KK)):
    for j in range(len(KK)):
        FILTER[i][j][0][0] = KK[i][j]

KERNEL2 = tf.constant(FILTER, dtype=tf.float32)


#######################################################################
#KERNEL 3  trial

FILTER = np.random.random_sample((5,5,1,1))   # SHARPER IMAGE filter input into conv2d:  rows, columns, input layers, output layers
factor = .1
KK = np.array(    [[0.,    0.,   0.,  0.,  0.], 
                   [0.,    0.,   0.,  0.,  0.],
                   [-factor,  -factor,  -factor, -factor, -factor],
                   [ factor,   factor,   factor,  factor,  factor],
                   [0.,    0.,   0.,  0.,  0.]])
               
for i in range(len(KK)):
    for j in range(len(KK)):
        FILTER[i][j][0][0] = KK[i][j]

KERNEL3 = tf.constant(FILTER, dtype=tf.float32)





strides = 1


#2   Construct model
x_out1 = tf.nn.conv2d(x, KERNEL1, strides=[1, strides, strides, 1], padding='SAME')  # SAME means pad with zeros => output same as input size
x_out2 = tf.nn.conv2d(x, KERNEL2, strides=[1, strides, strides, 1], padding='SAME')  # SAME means pad with zeros => output same as input size

x_out3 = tf.nn.conv2d(x, KERNEL3, strides=[1, strides, strides, 1], padding='SAME')  # SAME means pad with zeros => output same as input size


#3  Initializing the variables
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    
    out1 = sess.run(x_out1) #, feed_dict={x_K: K1})
    out2 = sess.run(x_out2) #, feed_dict={x_K: K1})
    out3 = sess.run(x_out3)
    
    
    FIG1 = np.zeros([M,N])
    FIG2 = np.zeros([M,N])
    FIG3 = np.zeros([M,N])
    for i in range(0,M):
      for j in range(0,N):
        FIG1[i][j] = out1[0][i][j][0]
        FIG2[i][j] = out2[0][i][j][0]
        FIG3[i][j] = out3[0][i][j][0]
        
    _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    

    #ax1.imshow(x1_test.reshape(28, 28), cmap=plt.cm.Greys, vmin = 0, vmax = 1); # cmap=plt.cm.Greys sees min(x) white, max(x) black
    #ax1.set_ylabel("y: 28 point")
    #ax1.set_xlabel("x: 28 point")
    #ax1.set_title("input figure")
    
    
    ax1.imshow(P1, cmap=plt.cm.Greys, vmin = 0, vmax = 1);
    ax1.set_title("Level 0 ")
    
    ax2.imshow(FIG1, cmap=plt.cm.Greys, vmin = -1, vmax = 1);
    ax2.set_title("FILTER 1")
    
    ax3.imshow(FIG2, cmap=plt.cm.Greys, vmin = -1, vmax = 1);
    ax3.set_title("FILTER 2")
    
    ax4.imshow(FIG3, cmap=plt.cm.Greys, vmin = -1, vmax = 1);
    ax4.set_title("FILTER 3")
    
    """
    ax1.imshow(P1, cmap=plt.cm.Greys, vmin = 0, vmax = 250);
    ax1.set_title("Level 0 ")
    
    ax2.imshow(FIG1, cmap=plt.cm.Greys, vmin = -250, vmax = 250);
    ax2.set_title("FILTER 1")
    
    ax3.imshow(FIG2, cmap=plt.cm.Greys, vmin = -250, vmax = 250);
    ax3.set_title("FILTER 2")
    
    ax4.imshow(FIG3, cmap=plt.cm.Greys, vmin = -250, vmax = 250);
    ax4.set_title("FILTER 3")
    """
    
    


# In[21]:

#19 try different kernels  on a parrot, bridge or a a simple black and white matrix
#    SAME AS #19 BUT USES FEED DICT TO INPUT DATA 
#   
# FREE PICTURES
#   http://www.freedigitalphotos.net/images/Birds_g52-Red_Macaw_p137333.html

# USE 16A to import data
import tensorflow as tf
import numpy as np


#print a sample first

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


from __future__ import print_function

#image = plt.imread("bridge.jpg")  
image = plt.imread("parrot.png")    # parrt.png is 360x412x4
#image = plt.imread("ID-100137333.jpg")




#imgplot = plt.imshow(image)
print(image.shape)
M = image.shape[0]
N = image.shape[1]
P1 = np.zeros([M,N])
for i in range(M):
      for j in range(N):
        P1[i][j] = image[i][j][0]  # CHOOSE LEVEL 0 OF 4 LAYERS THAT DEFINE COLOR AND ATTRIBUTES OF IMAGE
           

# TEST MY EDGE DETECTOR USING A SIMPLE MATRIX            
MAT = np.array(  [ [0,  0,  0,  0, 0, 0,  0,  0,  0, 0, 0,  0,  0,  0, 0], 
                   [0,  0,  0,  0, 0, 0,  0,  0,  0, 0, 0,  0,  0,  0, 0], 
                   [0,  0,  0,  0, 0, 0,  0,  0,  0, 0, 0,  0,  0,  0, 0],             
                   [0,  0,  0,  100, 100, 100,  100,  100,  100, 100, 0,  0,  0,  0, 0], 
                   [0,  0,  0,  100, 100, 100,  100,  100,  100, 100, 0,  0,  0,  0, 0], 
                   [0,  0,  0,  100, 100, 100,  100,  100,  100, 100, 0,  0,  0,  0, 0], 
                   [0,  0,  0,  100, 100, 100,  100,  100,  100, 100, 0,  0,  0,  0, 0], 
                   [0,  0,  0,  100, 100, 100,  100,  100,  100, 100, 0,  0,  0,  0, 0],
                   [0,  0,  0,  0, 0, 0,  0,  0,  0, 0,  0,  0,  0,  0, 0], 
                   [0,  0,  0,  0, 0, 0,  0,  0,  0, 0,  0,  0,  0,  0, 0]])
            

    
    
    
# NOW PREPARE the variable for THE TENSOR FLOW

MATRIX_or_IMAGE = 0  #  0 for simple matrix, 1 for image
if (MATRIX_or_IMAGE == 1):
    MAT = P1     # USE THIS TO PROCESS PARROT OR BRIDGE IMAGE
else:

    #P1 = MAT   # USE MATRIX AS INPUT COMMENT OUT THESE FOUR LINES TO USE ONE OF THE IMAGES (PARROT OR BRIDGE) INSTEAD IF SIMPLE MATRIX
    #x = MAT
    M = 10
    N = 15


""""
x = tf.convert_to_tensor(x,dtype=tf.float32)  # convert integer array to float32 tensor  (NOTE: (A) CONVERT TO TENSOR  (B) RECAST AS FLOAT32)
print('convert inout to float32 tensor: ', x)

#x = tf.reshape(x, shape=[-1, 1,M*N, 1])             #CONV needs a 4-dim (rank 4) input  DONT NEED THIS
#print(x)
x = tf.reshape(x, shape=[-1, M, N, 1])               # TENSORFLOW CONV COMMANDS NEED RANK OF TENSOR TO BE 4 (4 DIMENTIOANL TENSOR)
#  -1 is inferred such that total number of emelements does not change
print('convert tensor to rank 4 (4-dimentional tensor) for proper input to conv() command: ', x)

# THIS STATEMENT NEEDED FOR PROPER IMAGE PROCESSING FOR TENSOR FLOW FOR IMAGES,  BUT WHY? 
#x = tf.convert_to_tensor(x,dtype=tf.float64)              # could not convert using float32  REMOVE JUST FOR 10x10 example


#x = tf.cast(x, tf.float32)  # tensor flow code needs tensors in float32  (needed if origianl variable is FLOAT64 LIKE SOME IMAGES)
"""

# USE feed_dict to transfer data to sessin

#1- input data remains a numpy array
#2- but, its reshaped to have rank 4 as needed for TensorFlow CONV

#MAT = MAT.astype("float32")  # make sure data type is float32 as needed in TensorFlow

DATA = MAT.reshape(1,M,N,1)  # no longer a tensor

#3-  need to define x as a tensor
x = tf.placeholder("float32")


#1  define parameters and filter for proper input into conv2d

#######################################################################
#KERNEL 1  edge detect
FILTER = np.random.random_sample((5,5,1,1))   # filter input into conv2d:  rows, columns, input layers, output layers

KK = np.array(    [[0.,  0.,  0.,  0., 0.], 
                   [0.,  0.,  1.,  0., 0.],
                   [0.,  1., -4.,  1., 0.],
                   [0.,  0.,  1.,  0., 0.],
                   [0.,  0.,  0.,  0., 0.]])
KK = KK*5

for i in range(len(KK)):
    for j in range(len(KK)):
        FILTER[i][j][0][0] = KK[i][j]  # prepare the Kenel to meet TensorFlow convolusion command requirements  [data][data], [input layesr],[output la]
        
KERNEL1 = tf.constant(FILTER, dtype=tf.float32)

#######################################################################
#KERNEL 2  trial

FILTER = np.random.random_sample((5,5,1,1))   # SHARPER IMAGE filter input into conv2d:  rows, columns, input layers, output layers
KK = np.array(    [[0.,  -1.,  1.,  0., 0.], 
                   [0.,  -1.,  1.,  0., 0.],
                   [0.,  -1.,  1.,  0., 0.],
                   [0.,  -1.,  1.,  0., 0.],
                   [0.,  -1.,  1.,  0., 0.]])
               
KK = KK*10    
    
for i in range(len(KK)):
    for j in range(len(KK)):
        FILTER[i][j][0][0] = KK[i][j]

KERNEL2 = tf.constant(FILTER, dtype=tf.float32)


#######################################################################
#KERNEL 3  trial

FILTER = np.random.random_sample((5,5,1,1))   # SHARPER IMAGE filter input into conv2d:  rows, columns, input layers, output layers
factor = .1
KK = np.array(    [[0.,    0.,   0.,  0.,  0.], 
                   [0.,    0.,   0.,  0.,  0.],
                   [-factor,  -factor,  -factor, -factor, -factor],
                   [ factor,   factor,   factor,  factor,  factor],
                   [0.,    0.,   0.,  0.,  0.]])
               
for i in range(len(KK)):
    for j in range(len(KK)):
        FILTER[i][j][0][0] = KK[i][j]

KERNEL3 = tf.constant(FILTER, dtype=tf.float32)





strides = 1


#2   Construct model
x_out1 = tf.nn.conv2d(x, KERNEL1, strides=[1, strides, strides, 1], padding='SAME')  # SAME means pad with zeros => output same as input size
x_out2 = tf.nn.conv2d(x, KERNEL2, strides=[1, strides, strides, 1], padding='SAME')  # SAME means pad with zeros => output same as input size

x_out3 = tf.nn.conv2d(x, KERNEL3, strides=[1, strides, strides, 1], padding='SAME')  # SAME means pad with zeros => output same as input size


#3  Initializing the variables
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    
    out1 = sess.run(x_out1, feed_dict={x: DATA})
    out2 = sess.run(x_out2, feed_dict={x: DATA})
    out3 = sess.run(x_out3, feed_dict={x: DATA})
    
    
    FIG1 = np.zeros([M,N])
    FIG2 = np.zeros([M,N])
    FIG3 = np.zeros([M,N])
    for i in range(0,M):
      for j in range(0,N):
        FIG1[i][j] = out1[0][i][j][0]
        FIG2[i][j] = out2[0][i][j][0]
        FIG3[i][j] = out3[0][i][j][0]
        
    _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    

    #ax1.imshow(x1_test.reshape(28, 28), cmap=plt.cm.Greys, vmin = 0, vmax = 1); # cmap=plt.cm.Greys sees min(x) white, max(x) black
    #ax1.set_ylabel("y: 28 point")
    #ax1.set_xlabel("x: 28 point")
    #ax1.set_title("input figure")
    
    
    ax1.imshow(MAT, cmap=plt.cm.Greys, vmin = 0, vmax = 1);
    ax1.set_title("Level 0 ")
    
    ax2.imshow(FIG1, cmap=plt.cm.Greys, vmin = -1, vmax = 1);
    ax2.set_title("FILTER 1")
    
    ax3.imshow(FIG2, cmap=plt.cm.Greys, vmin = -1, vmax = 1);
    ax3.set_title("FILTER 2")
    
    ax4.imshow(FIG3, cmap=plt.cm.Greys, vmin = -1, vmax = 1);
    ax4.set_title("FILTER 3")
    
    """
    ax1.imshow(P1, cmap=plt.cm.Greys, vmin = 0, vmax = 250);
    ax1.set_title("Level 0 ")
    
    ax2.imshow(FIG1, cmap=plt.cm.Greys, vmin = -250, vmax = 250);
    ax2.set_title("FILTER 1")
    
    ax3.imshow(FIG2, cmap=plt.cm.Greys, vmin = -250, vmax = 250);
    ax3.set_title("FILTER 2")
    
    ax4.imshow(FIG3, cmap=plt.cm.Greys, vmin = -250, vmax = 250);
    ax4.set_title("FILTER 3")
    """
    
    


# In[34]:

print(P1.shape)
print(np.min(P1))
print(np.max(P1))


# In[37]:

print(np.min(FIG1))
print(np.max(FIG1))
print(np.min(FIG2))
print(np.max(FIG2))
print(np.min(FIG3))
print(np.max(FIG3))


# In[5]:

MAT


# In[12]:

MAT1 = MAT.reshape(1,10,15,1)
print(MAT1.shape)


# In[ ]:



