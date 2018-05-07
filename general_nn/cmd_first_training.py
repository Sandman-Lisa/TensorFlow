# -*- coding: utf-8 -*-

from __future__ import print_function
import gnn_methods as gm
import matplotlib.pyplot as mp
import sys
trainning_task_file         = 'train_task.cfg'
trainning_input_file        = 'train_input.txt'
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
            trainning_input_file = sys.argv[it+1]
            print('input_file :',trainning_input_file)
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
           python cmd_first_trainning.py 
               -t \\path\\to\\任务配置.cfg
               -i \\path\\to\\训练样本.txt
               -m \\path\\to\\结果文件夹
    
           """       
    )
    sys.exit(0)

result = gm.train_model(trainning_task_file,trainning_input_file,model_path,1)
        
#mp.plot(result["x_plot"],result["y_plot"],'b')
