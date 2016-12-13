# G3/X  LQR  using TENSORFLOW  and AdamOptimizer

# GOAL:  find the best linear fit for a group of data


# Author:  Ramin Monajemy
# 12/12/16




import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
%matplotlib inline


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
%matplotlib inline

print('*** START 2 ***')

DATA_c = 1  # Choose "0" for 6 point pre-chosen data,  "1" for 100 point random data

if DATA_c == 1:
    #A   DEFINE DATA
    # Experiment 2:  define a larger group of data points
    NN = 100  # number of data pairs
    DATA = np.zeros([100,2])
    for j in range(100):
        DATA[j][0]= np.random.normal(0.0, 10)
        DATA[j][1]= DATA[j][0] * 2.0 + 5.0 + np.random.normal(0.0, 20.0)
    ALPHA = 1.0

# Experiment 1: just define 6 data points


if DATA_c == 0:
    DATA = [[0.5,0],
            [1,4],
            [2.1,6.6],
            [3,10],
            [4.1,15],
            [5.5,15]]

    ALPHA = 0.3   

       
   
batch_x = np.zeros(len(DATA))
batch_y = np.zeros(len(DATA))

for i in range(0,len(DATA)):
    batch_x[i] = DATA[i][0]
    batch_y[i] = DATA[i][1]    
    
    
# fit equation:    Y = a X + b

ITERATIONS = 100
Ndata = len(DATA)



LOSS_SAVE = np.zeros([ITERATIONS,1])

  
    
#B DEFINE ALL TENSORS

a = tf.Variable(30.0)
b = tf.Variable(10.0)



LOSS = tf.Variable(0.0)


#x = tf.Variable(0.0)
#y = tf.Variable(0.0)
x = tf.placeholder("float")  # MUST USE PLACEHOLDER TO AVOID ERRORS EVEN FOR VERY BASIC CODE. BUT, WHY?
y = tf.placeholder("float")  #  (THAT JUST MULTIPLIES SINGLE PARAMETRS)

#C  DEFINE GRAPH 


LOSS_V = ( y   - (a*x + b) )
#LOSS_Vt = tf.transpose(LOSS_V)
LOSS_V2  = LOSS_V*LOSS_V  # 12/12/16 this just still yields a vector
#LOSS = tf.matmul(LOSS_Vt,2.0)  # this yields tensor not available error
LOSS = tf.reduce_sum(LOSS_V2) # this works


optimizer = tf.train.AdamOptimizer(learning_rate=ALPHA).minimize(LOSS)

a_SAVE = 0.0
b_SAVE = 0.0

#D  define init function
init = tf.initialize_all_variables()

#E LAUNCH THE GRAPH
with tf.Session() as sess:
    sess.run(init)
    LOSS_S = 0.0
        
    #   LOSS =  xLOSS + ( Yi   - (a*Xi + b) )**2    - LOSS EXPRESSION FOR REFERENCE    
    #   a = xa - ALPHA *  dL_da                      - EXPRESSION TO CALCULATE PARAMETERS A AND B

   
            
    for k in range(ITERATIONS):
     
        
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        LOSS_out = sess.run(LOSS, feed_dict={x: batch_x,y: batch_y})
        #print("******** LOSS = ", LOSS_out)
        #print('LOSS_SAVE = ', LOSS_SAVE)
        LOSS_SAVE[k] = LOSS_out

       
        a_SAVE = sess.run(a)
        b_SAVE = sess.run(b)

        print('a_SAVE =',a_SAVE, ' b_SAVE =',b_SAVE)
       
       


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





print('*** END 2 ***')
