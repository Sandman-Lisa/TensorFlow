# -*- coding: utf-8 -*-

from __future__ import print_function
import gnn_methods as gm
import matplotlib.pyplot as mp
import sys
trainning_task_file         = 'train_task.cfg'
testing_file                = 'test_set.txt'
model_path                  = './saved_model/'

hit_arg = 0;
totalLen = len(sys.argv)
for it in range(totalLen):
    if sys.argv[it].startswith('-t') :
        if it < totalLen - 1:
            trainning_task_file = sys.argv[it+1]
            print('task_file :',trainning_task_file)
            hit_arg = hit_arg + 1
    if sys.argv[it].startswith('-i') :
        if it < totalLen - 1:
            testing_file = sys.argv[it+1]
            print('input_file :',testing_file)
            hit_arg = hit_arg + 1
    if sys.argv[it].startswith('-m') :
        if it < totalLen - 1:
            model_path = sys.argv[it+1]
            print('model_path :',model_path)
            hit_arg = hit_arg + 1
    if sys.argv[it].startswith('-h') :
            hit_arg = -3

if hit_arg < 3:
    print ("""
           训练神经网络
           python cmd_test_model.py 
               -t \\path\\to\\任务配置.cfg
               -i \\path\\to\\测试样本.txt
               -m \\path\\to\\模型文件夹
    
           """       
    )
    sys.exit(0)

result = gm.test_model(trainning_task_file,testing_file,model_path)

#mp.plot(result["x_plot"],result["y_plot"],'b')