# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 20:48:53 2016

@author: Ramin
"""



# GOALS
# G2- implement a linear curve fit using Gradient Descent for 6 random group of data, using quadratic linear regulator in Tensorflow
#  Prepared and Debugged by Ramin Monajemy
#  START:  10/31/16




import numpy as np
import matplotlib.pyplot as plt
import time
get_ipython().magic(u'matplotlib inline')





#A   DEFINE DATA
"""
# Experiment 2:  define a larger group of data points
NN = 100  # number of data pairs
DATA = np.zeros([100,2])
for j in range(100):
    DATA[j][0]= np.random.normal(0.0, 10)
    DATA[j][1]= DATA[j][0] * 2.0 + 5.0 + np.random.normal(0.0, 20.0)

"""

# Experiment 1: just define 6 data points

DATA = [[0.5,0],
        [1,4],
        [2.1,6.6],
        [3,10],
        [4.1,15],
        [5.5,15]]
   
    
# fit equation:    Y = a X + b

ITERATIONS = 10
Ndata = len(DATA)

ALPHA = 0.01

LOSS_SAVE = np.zeros([ITERATIONS,1])

#B DEFINE ALL TENSORS

#a = tf.Variable(tf.random_normal([1]))
#b = tf.Variable(tf.random_normal([1]))

a = tf.Variable(1.0)
b = tf.Variable(2.0)

xa = tf.Variable(0.0)
xb = tf.Variable(0.0)

xxa = tf.Variable(0.0)
xxb = tf.Variable(0.0)

#a = tf.placeholder("float32")
#b = tf.placeholder("float32")

"""
LOSS = tf.placeholder("float32")



Xi = tf.placeholder("float32")
Yi = tf.placeholder("float32")

dL_da = tf.placeholder("float32")
dL_db = tf.placeholder("float32")
"""

LOSS = tf.Variable(0.0)
xLOSS = tf.Variable(0.0)  # use the  x variables here, and below, to 
                          # implement recursive loss calculation


Xi = tf.Variable(0.0)
Yi = tf.Variable(0.0)

dL_da = tf.Variable(0.0)
dL_db = tf.Variable(0.0)
xdL_da = tf.Variable(0.0)
xdL_db = tf.Variable(0.0)



#C  DEFINE GRAPH 
LOSS =  xLOSS + ( Yi   - (xxa*Xi + xxb) )**2  # cant use loss = loss + xxx,  the right side loss will always be initalized to above value
#LOSS = Xi * Yi 
  
#TEST = xxa * 2    
    
dL_da = xdL_da - 2 * Xi * ( Yi   - (xxa*Xi + xxb) ) 
dL_db = xdL_db - 2 * ( Yi   - (xxa*Xi + xxb) )

    
    
a = xa - ALPHA *  dL_da 
b = xb - ALPHA *  dL_db 


a_SAVE = 0.0
b_SAVE = 0.0

#D  define init function
init = tf.initialize_all_variables()

#E LAUNCH THE GRAPH
with tf.Session() as sess:
    sess.run(init)
    
    #tf.a = 5.0
    #tf.b = 5.0
    
   
    
    for k in range(ITERATIONS):  #  RECORD IN NOTEBOOK =  must use tf.x format for any variable being modified in inside session
        
        dL_da_SAVE = 0.0  # set all "sums" to zero for each iteration
        dL_db_SAVE = 0.0

        LOSS_S = 0.0
        
    
        #   LOSS =  xLOSS + ( Yi   - (a*Xi + b) )**2     LOSS EXPRESSION FOR REFERENCE    
        #   a = xa - ALPHA *  dL_da                      EXPRESSION TO CALCULATE PARAMETERS A AND B
    
        for i in range(Ndata): #N data points # calcualte value of loss function, differentaial of loss wrt a and b over all data points
            xi_data = DATA[i][0]
            yi_data = DATA[i][1]       
            
            LOSS_S = sess.run(LOSS, feed_dict={xLOSS: LOSS_S, Xi: xi_data, Yi: yi_data, xxa: a_SAVE, xxb: b_SAVE}) 
            #print('LOSS_SAVE = ', LOSS_SAVE)
            LOSS_SAVE[k] = LOSS_S
            dL_da_SAVE = sess.run( dL_da, feed_dict={xdL_da: dL_da_SAVE, Xi: xi_data, Yi: yi_data, xxa: a_SAVE, xxb: b_SAVE})
            dL_db_SAVE = sess.run( dL_db, feed_dict={xdL_db: dL_db_SAVE, Xi: xi_data, Yi: yi_data, xxa: a_SAVE, xxb: b_SAVE})
            #print('dL_da_SAVE', dL_da_SAVE)
            #LOSS_SAVE[k] = L  

           
        a_SAVE = sess.run(a, feed_dict={dL_da: dL_da_SAVE, xa: a_SAVE})
        b_SAVE = sess.run(b, feed_dict={dL_db: dL_db_SAVE, xb: b_SAVE})

        #print('dL_da_SAVE', dL_da_SAVE)
        #print('a_SAVE =',a_SAVE)
        #print('b_SAVE =',b_SAVE)
        #print('LOSS =',LOSS_S)
        #print('a =',a_SAVE)
        #print('b =',b_SAVE)

        #TEST_SAVE = sess.run(TEST, feed_dict={xxa: a_SAVE})     
        #print('TEST_SAVE = 2 * a = ', TEST_SAVE) 
        
        
    DATA_X = np.zeros(len(DATA))
    DATA_Y = np.zeros(len(DATA))
    DATA_Yp = np.zeros(len(DATA))

    for i in range(len(DATA)):
        DATA_X[i] = DATA[i][0]
        DATA_Y[i] = DATA[i][1]
        DATA_Yp[i]  = a_SAVE * DATA_X[i] + b_SAVE


    plt.figure(1)
    plt.subplot(211)
    plt.plot(DATA_X, DATA_Y,'gx',  DATA_X, DATA_Yp)   
    #plt.axis([-1, 6, -1, 20])
    #plt.ylabel('Y')
    #plt.xlabel('X')
    #plt.title('path smoothing')
    plt.grid(True)
    plt.show()  

    plt.subplot(212)
    plt.plot(LOSS_SAVE,'k')   
    #plt.axis([-1, 6, -1, 20])
    #plt.ylabel('Y')
    plt.xlabel('iteration')
    plt.title('LOSS')
    plt.grid(True)
    plt.show()  
        

