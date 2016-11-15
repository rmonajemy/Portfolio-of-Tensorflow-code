
#  10 digit classification project using Convolutional Neural Networks

# learned how to program it, how it works, did fine tuning, added plots for verification and demo...  by Ramin Monajemy

# used code at this location for reference:
##  https://github.com/aymericdamien/TensorFlow-Examples



from __future__ import print_function


print('***  start download  ****')
# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

print('***  download done  ****')

print('MNIST DATA train images: ', mnist.train.images.shape)
print('MNIST DATA train labels: ', mnist.train.labels.shape)
print('MNIST DATA validation images: ', mnist.validation.images.shape)
print('MNIST DATA validation labels: ', mnist.validation.labels.shape)
print('MNIST DATA test images: ', mnist.test.images.shape)
print('MNIST DATA test labels: ', mnist.test.labels.shape)



# In[6]:

print('MNIST DATA train number of data: ', mnist.train.num_examples)
print('MNIST DATA validation number of data: ', mnist.validation.num_examples)
print('MNIST DATA test number of data: ', mnist.test.num_examples)


# In[26]:


# 11B    Verified perceptron to identify 10 digits

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
get_ipython().magic(u'matplotlib inline')



#11x  Verify that the perceptron actually worked on a character

#N = 106
#x1_test = mnist.test.images[N][:]

#_, (ax1) = plt.subplots(1, 1)
#ax1.imshow(x1_test.reshape(28, 28), cmap=plt.cm.Greys);
#ax1.set_ylabel("y: 28 point")
#ax1.set_xlabel("x: 28 point")
#ax1.set_title("PLOT Nth CHARACTER")



# NOW START THE MAIN TRAINING EVENT
# (8/19/16)  best configuration: (all using AdamOptimizer which seems to perform better than GradientDescentOptimizer.  WHY?)
  #a- 2 layer, 32, 32, learning rate = 0.01,  no-nonlinarity,  49 seconds => 87% in 10 steps 
  #b- 2 layer, 32, 32, learning rate = 0.01,  relu, 42 seconds =>            92% to  94% in 10 steps 
  #c- 2 layer, 32, 32, learning rate = 0.01, tanh, 40 seconds =>            92% in 10 steps
  #d- 2 layer, 32, 32, learning rate = 0.01, sigmoid, 45 seconds =>         95% to 96% in 10 steps
  #   //                                                                      94% in just 3 steps
# Parameters
#learning_rate = 0.001  # ORIGINAL
learning_rate = 0.01  
training_steps = 1
batch_size = 100
display_step = 1

loss_vector =  np.zeros([1,training_steps])  # store loss at each step

# Network Parameters  # (8/19/16) original setup resulted in 95% accuracy
#n_hidden_1 = 256 # 1st layer number of features  ORIGINAL
#n_hidden_2 = 256  # 2nd layer number of features ORIGINAL
n_hidden_1 = 32 # 1st layer number of features
n_hidden_2 = 32 # 2nd layer number of features

n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

OUT_1 = tf.placeholder("float", [None, n_input])


start_time = time.time()  # for model

#///////////////////////////////////////////////////////////////////////
#//////////////  ORIGINAL MODEL - 2 LAYERS, RELU ///////////////////////
#///////////////////////////////////////////////////////////////////////
# Create model

def multilayer_perceptron(x, weights, biases):
    # Hidden layer with non linear activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])   # (X = 1 x 784) x (784 x 32)  + (1x32)  = [1 x32]
    #layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.sigmoid(layer_1)
    #layer_1 = tf.nn.tanh(layer_1) 

    # Hidden layer with non linear activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])  # [layer_1 = (1 x32)]  x (32 x 32) + (1 x 32)  = 1 x32
    #layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.sigmoid(layer_2)
    #layer_2 = tf.nn.tanh(layer_2)
    
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']     #  1 x 32  * 32 x 10  +  1 x 10  = 1 x  10
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

#///////////////////////////////////////////////////////////////////////

"""
#///////////////////////////////////////////////////////////////////////
#//////////////   B- 1 LAYER, RELU ///////////////////////
#///////////////////////////////////////////////////////////////////////
# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    
    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

#///////////////////////////////////////////////////////////////////////

"""


# Construct model
pred = multilayer_perceptron(x, weights, biases)  # if batch size of X = "100" x 784 (as is done later in  each step later);
                                                  # then pred = 100 x (1X10) logit (largest number of vector is the prediction)
#  softmax will convert them to exp(x)/ (sum(exp(x)))  => probability

# MY OWN STATEMENT FOR VERIFICATION PURPOSES, RUN IN THE LOOP USING SESS RUN
ARGMAX_Y1 =  tf.argmax(pred, 1)
        
# Define loss and optimizer

#1 original loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))     # ORIGINAL 
#                                                                            # note: y  = 100 x (1X10) original correct 1-hot vectors
# regularization (add sum of square of gains to penalize high gains)
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))  + 0.001*tf.nn.l2_loss(weights['h1'])   # REGULARIZATION 
#  just adding l2_loss for one of three weights if graph tiples the time requirement


#2 add l2 loss to penalize large gains
#  nn.l2_loss(t)

#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # ORIGINAL

# Initializing the variables
init = tf.initialize_all_variables()

end_time = time.time()  # for model
print('***  Model construction time (s): ', end_time-start_time)
#print("***  model construct time (s): ", '%04d' % end_time-start_time)  DID NOT WORK
# print("Step:", '%04d' % (epoch+1)

start_time = time.time()  # for loop
# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_steps):   # epoch to 0 to 9
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)  #  55000/100  = 550 batches of data (each input together)
        # Loop over all batches
        for i in range(total_batch):  # 550
            batch_x, batch_y = mnist.train.next_batch(batch_size)  #  every call to mnist.train.next_batch(100) loads next batch of 100
            # Run optimization op (backprop) and cost op (to get loss value)
            # batch_x is  100 x 784  (784 characters), and batch_y is the associated 1-hot  100 x 10 vector for each line 
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:  #  TRUE when remainder is 0;  every step for display_step = 1,  every other step for 2, ...
            print("Step:", '%04d' % (epoch+1), "cost=",                 "{:2.2f}".format(avg_cost))
        #loss_vector[0,epoch] = c  # PLOT OF COST IS VERY ERRATIC
        loss_vector[0,epoch] = avg_cost

    print("Optimization Finished!")
    end_time = time.time()  # for model
    print('***  Training Time (s): ', end_time-start_time)
    
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    
    
    
    # different ways to use above TensorFlow
    
   
    N = 400
    # feed three characters to the perceptron, verify detection and plot the them
    ARGMAX_Y =  tf.argmax(pred, 1)
    #Method ONE: use sess run
    TEST_SESS = sess.run([ARGMAX_Y1], feed_dict={x: mnist.test.images[N:N+4][:]})
    print('I. ARGMAX FOR FEW CHARACTER INPUTS USING SESS RUN:', TEST_SESS)

    #Method TWO: put it in a variable and print it  (pred outside of sess loop, but argmax_y equation being inside)
    TEST = ARGMAX_Y.eval({x: mnist.test.images[N:N+4][:]})
    print('II. ARGMAX FOR FEW CHARACTER INPUTS USING .EVAL:', TEST)
    
          
    
    
    x1_test = mnist.test.images[N][:]
    x2_test = mnist.test.images[N+1][:]
    x3_test = mnist.test.images[N+2][:]
    x4_test = mnist.test.images[N+3][:]
    #OneHot = tf.equal(tf.argmax(OUT_1, 1))
    # Calculate accuracy
    
    #print("One Hot:", OneHot.eval({OUT_1: OUT}))
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

    ax1.imshow(x1_test.reshape(28, 28), cmap=plt.cm.Greys);
    #ax1.set_ylabel("y: 28 point")
    #ax1.set_xlabel("x: 28 point")
    #ax1.set_title("PLOT Nth CHARACTER")
    
    ax2.imshow(x2_test.reshape(28, 28), cmap=plt.cm.Greys);
    ax3.imshow(x3_test.reshape(28, 28), cmap=plt.cm.Greys);
    ax4.imshow(x4_test.reshape(28, 28), cmap=plt.cm.Greys);
    


