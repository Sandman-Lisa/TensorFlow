# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as mp
trainning_input_file        = 'train_input.txt'
testing_file = 'test_set.txt'

m=102400
n=2
K=3
#generate a circle region at center and size is R
center1 = np.random.rand(2).astype(np.float32)/4 + 0.2
center2 = -1 * np.random.rand(2).astype(np.float32)/4 - 0.5
r1 = np.random.rand(1).astype(np.float32) * 0.4 + 0.8
r2 = np.random.rand(1).astype(np.float32) * 0.1 + 0.2

x_data = np.random.rand(m,n).astype(np.float32) * 2 - 1
#calc the y for each training sets
y_data = np.zeros([m,K]).astype(np.float32)
for idx in range(m):
    if 1.2*(x_data[idx,0] - center1[0])**2 + (x_data[idx,1] - center1[1])**2 <= r1**2 *np.cos(x_data[idx,1]*6)*np.sin(x_data[idx,0]*12):
        y_data[idx,1] = 1
    elif (x_data[idx,0] - center2[0])**2  + (x_data[idx,1] - center2[1])**2 <= r2 **x_data[idx,0]-r2 **x_data[idx,1]:
        y_data[idx,2] = 1
    else:
        y_data[idx,0] = 1

mp.plot(x_data[y_data[:,0]==1,0],x_data[y_data[:,0]==1,1],'r.')
mp.plot(x_data[y_data[:,1]==1,0],x_data[y_data[:,1]==1,1],'g.')
mp.plot(x_data[y_data[:,2]==1,0],x_data[y_data[:,2]==1,1],'b.')

with open(trainning_input_file, 'wt') as f:
    for idx in range(m):
        #print('%10f,%10f,%1d,%1d,%1d'%(x_data[idx][0],x_data[idx][1],y_data[idx][0],y_data[idx][1],y_data[idx][2]),file = f)
        print('%10f,%10f,%10f,%10f,%10f,%10f,%10f,%10f,%10f,%1d,%1d,%1d'%(x_data[idx][0],x_data[idx][1],x_data[idx][1]*x_data[idx][0],x_data[idx][1]**2,x_data[idx][0]**2,np.exp(x_data[idx][1]),np.exp(x_data[idx][0]),np.cos(x_data[idx][0]),np.cos(x_data[idx][1]),y_data[idx][0],y_data[idx][1],y_data[idx][2]),file = f)


for idx in range(m):
    if 1.2*(x_data[idx,0] - center1[0])**2 + (x_data[idx,1] - center1[1])**2 <= r1**2 *np.cos(x_data[idx,1]*6)*np.sin(x_data[idx,0]*12):
        y_data[idx,1] = 1
    elif (x_data[idx,0] - center2[0])**2  + (x_data[idx,1] - center2[1])**2 <= r2 **x_data[idx,0]-r2 **x_data[idx,1]:
        y_data[idx,2] = 1
    else:
        y_data[idx,0] = 1

with open(testing_file, 'wt') as f:
    for idx in range(m):
        #print('%10f,%10f,%1d,%1d,%1d'%(x_data[idx][0],x_data[idx][1],y_data[idx][0],y_data[idx][1],y_data[idx][2]),file = f)
        print('%10f,%10f,%10f,%10f,%10f,%10f,%10f,%10f,%10f,%1d,%1d,%1d'%(x_data[idx][0],x_data[idx][1],x_data[idx][1]*x_data[idx][0],x_data[idx][1]**2,x_data[idx][0]**2,np.exp(x_data[idx][1]),np.exp(x_data[idx][0]),np.cos(x_data[idx][0]),np.cos(x_data[idx][1]),y_data[idx][0],y_data[idx][1],y_data[idx][2]),file = f)
