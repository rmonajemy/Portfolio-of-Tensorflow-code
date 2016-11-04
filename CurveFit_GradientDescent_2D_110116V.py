
# G1/3  LQR  using Python  (VERIFIED 11/1/16)

#  START:  10/31/16

# GOALS
# G1- implement a linear curve fit for 6 XY data point using quadratic linear regulator in Python first, then in TensorFlow
# G2- figure out how to store linear gains and use outside of of main Tensorflow session
# G3- then replace optimize with following standard tools and run comparisons and experiments, and figure out how they work
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#     optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
%matplotlib inline


# Experiment 1: just define 6 data points
"""
DATA = [[0.5,0],
        [1,4],
        [2.1,6.6],
        [3,10],
        [4.1,15],
        [5.5,15]]
"""


# Experiment 2:  define a larger group of data points
NN = 100
DATA = np.zeros([100,2])
for j in range(100):
    DATA[j][0]= np.random.normal(0.0, 10)
    DATA[j][1]= DATA[j][0] * 2.0 + 5.0 + np.random.normal(0.0, 20.0)


# fit equation:    Y = a X + b

ITERATIONS = 50
N = len(DATA)
a = 1 # init values for a , b
b = 1

LOSS_SAVE = np.zeros([ITERATIONS,1])


ALPHA = 0.00001

for k in range(ITERATIONS):
    dL_da = 0  # set all "sums" to zero for each iteration
    dL_db = 0
    
    LOSS = 0
    
    for i in range(N): #N data points # calcualte value of loss function, differentaial of loss wrt a and b over all data points

        Xi  = DATA[i][0]
        Yi  = DATA[i][1]


        LOSS +=  ( Yi   - (a*Xi + b) )**2
    
        
        dL_da += -2 * Xi * ( Yi   - (a*Xi + b) ) 
        dL_db += -2 * ( Yi   - (a*Xi + b) )
        
    LOSS_SAVE[k] = LOSS  
    
    a = a - ALPHA *  dL_da 
    b = b - ALPHA *  dL_db 


print('a',a)
print('b',b)


DATA_X = np.zeros(len(DATA))
DATA_Y = np.zeros(len(DATA))
DATA_Yp = np.zeros(len(DATA))

for i in range(len(DATA)):
    DATA_X[i] = DATA[i][0]
    DATA_Y[i] = DATA[i][1]
    DATA_Yp[i]  = a * DATA_X[i] + b


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



