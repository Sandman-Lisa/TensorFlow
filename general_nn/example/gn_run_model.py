# -*- coding: utf-8 -*-

from __future__ import print_function
import gnn_methods as gm
import matplotlib.pyplot as mp
trainning_task_file         = 'train_task.cfg'
input_file                  = 'test_set.txt'
output_file                 = 'result.txt'
model_path                  = './saved_model/'
ret = gm.run_model(trainning_task_file,input_file,model_path,output_file)
x_test = ret["x_test"]
result = ret["result"]

mp.plot(x_test[result[:,0]**2>(result[:,1]+result[:,2])**2,0],x_test[result[:,0]**2>(result[:,1]+result[:,2])**2,1],'r.')
mp.plot(x_test[result[:,1]**2>(result[:,0]+result[:,2])**2,0],x_test[result[:,1]**2>(result[:,0]+result[:,2])**2,1],'g.')
mp.plot(x_test[result[:,2]**2>(result[:,1]+result[:,0])**2,0],x_test[result[:,2]**2>(result[:,1]+result[:,0])**2,1],'b.')
mp.show()
dp_all = gm.dump_all(trainning_task_file,model_path)

