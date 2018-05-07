# -*- coding: utf-8 -*-

from __future__ import print_function
import gnn_methods as gm
import matplotlib.pyplot as mp
trainning_task_file         = 'train_task.cfg'
trainning_input_file        = 'train_input.txt'
model_path                  = './saved_model/'

result = gm.train_model(trainning_task_file,trainning_input_file,model_path,0)
        
mp.plot(result["x_plot"],result["y_plot"],'b')

