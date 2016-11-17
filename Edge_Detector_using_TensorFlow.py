

# In[11]:


#  Edge Detector in Tensorflow

#1-  peripehry
#2-  horizonal edges
#3-  vertical edges

#prepared and debugged by:  Ramin Monajemy



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
    
    
