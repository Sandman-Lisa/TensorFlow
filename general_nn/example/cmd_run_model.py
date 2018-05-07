# -*- coding: utf-8 -*-

from __future__ import print_function
import gnn_methods as gm
import matplotlib.pyplot as mp
import sys
trainning_task_file         = 'train_task.cfg'
input_file                  = 'test_set.txt'
output_file                 = 'result.txt'
model_path                  = './saved_model/'

hit_arg = 0
totalLen = len(sys.argv)
for it in range(totalLen):
    if sys.argv[it].startswith('-t') :
        if it < totalLen - 1:
            trainning_task_file = sys.argv[it+1]
            print('task_file :',trainning_task_file)
            hit_arg = hit_arg + 1
    if sys.argv[it].startswith('-i') :
        if it < totalLen - 1:
            input_file = sys.argv[it+1]
            print('input_file :',input_file)
            hit_arg = hit_arg + 1
    if sys.argv[it].startswith('-m') :
        if it < totalLen - 1:
            model_path = sys.argv[it+1]
            print('model_path :',model_path)
            hit_arg = hit_arg + 1
    if sys.argv[it].startswith('-o') :
        if it < totalLen - 1:
            output_file = sys.argv[it+1]
            print('output_file :',output_file)
            hit_arg = hit_arg + 1
    if sys.argv[it].startswith('-h') :
            hit_arg = -3

if hit_arg < 4:
    print ("""
           运神经网络任务
           python cmd_run_model.py 
               -t \\path\\to\\任务配置.cfg
               -i \\path\\to\\测试样本.txt
               -m \\path\\to\\模型文件夹
               -o \\path\\to\\运行结果.txt
    
           """       
    )
    sys.exit(0)

result = gm.run_model(trainning_task_file,input_file,model_path,output_file)
