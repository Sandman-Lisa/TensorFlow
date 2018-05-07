# -*- coding: utf-8 -*-

from __future__ import print_function
import gnn_methods as gm
import matplotlib.pyplot as mp
trainning_task_file         = 'train_task.cfg'
testing_file                = 'test_set.txt'
model_path                  = './saved_model/'

result = gm.test_model(trainning_task_file,testing_file,model_path)

mp.plot(result["x_plot"],result["y_plot"],'b')