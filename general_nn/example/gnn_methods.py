# -*- coding: utf-8 -*-

from __future__ import print_function
from tensorflow import summary as tfs
import configparser
import re
import tensorflow as tf
import numpy as np

def create_graph(n,K,dlambda,ns_hidden):  
    """
    create_graph函数用来创建神经网络
    @para n 输入特征数
    @para K 输出向量维度
    @para dlambda 正则化小数 0.00001
    @para ns_hidden [第一隐层数目，第二隐层数目,...最后一个隐层数目 ]
    @return 图,可用get_tensor_by_name等函数获得其中的各类结构，主要结构：
       含义                    类型    名称
    1. 输入向量(占位符)        tensor  network/input:0
    2. lambda 正则化因子       tensor  network/regular:0
    3. 线性转移矩阵            tensor  network/W1:0 ~ Wm:0, m为层数。
    4. 线性转移偏执            tensor  network/b1:0 ~ bm:0, m为层数。
    5. 线性转移结果z=aW+b      tensor  network/z1:0 ~ zm:0, m为层数。
    6. 各隐层输出  a=f(z)      tensor  network/a1:0 ~ am:0, m为层数。
    7. 输出向量                tensor  network/output:0
    8. 训练样本参考(占位符)    tensor  loss/tr_out:0
    9. 网络代价                tensor  loss/loss:0，不包括正则化
    10.训练器               operation  train/train
    """
    ns_array = ns_hidden[:]
    #Output is the last layer, append to last
    ns_array.append(K)
    hidden_layer_size = len(ns_array)
    #--------------------------------------------------------------
    #create graph
    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope('network'):
            s = [n]
            a = [tf.placeholder(tf.float32,[None,s[0]],name="input")]
            W = []
            b = []
            z = []
            punish = tf.constant(0.0, name='regular')
            for idx in range(0,hidden_layer_size)    :
                s.append(int(ns_array[idx]))
                W.append(tf.Variable(tf.random_uniform([s[idx],s[idx+1]],0,1),\
                                     name='W'+str(idx+1)))
                b.append(tf.Variable(tf.random_uniform([1,s[idx+1]],0,1),\
                                     name='b'+str(idx+1)))
                z.append(tf.add(tf.matmul(a[idx],W[idx]) , b[idx],\
                                name='z'+str(idx+1)))
                if (idx < hidden_layer_size - 1):
                    a_name = 'a'+str(idx+1)
                else:
                    a_name = 'output'
                a.append(tf.nn.tanh(z[idx],name=a_name))
                punish = punish + tf.reduce_sum(W[idx]**2) * dlambda
                
                tfs.histogram('W'+str(idx+1),W[idx])
                tfs.histogram('b'+str(idx+1),b[idx])
                tfs.histogram('a'+str(idx+1),a[idx+1])
        #--------------------------------------------------------------
        with tf.name_scope('loss'):
            y_ = tf.placeholder(tf.float32,[None,K],name="tr_out")
            pure_loss = tf.reduce_mean(tf.square(a[hidden_layer_size]-y_),\
                                       name="pure_loss")
            loss = tf.add(pure_loss, punish, name="loss")
            tfs.scalar('loss',loss)
            tfs.scalar('punish',punish)
            tfs.scalar('pure_loss',pure_loss)
        with tf.name_scope('train'):        
            optimizer = tf.train.AdamOptimizer(name="optimizer")
            optimizer.minimize(loss,name="train")  
            #记录网络被训练了多少样本次
            train_times = tf.Variable(tf.zeros([1,1]), name='train_times')            
            train_trunk = tf.placeholder(tf.float32,[None,1],\
                                         name="train_trunk")
            tf.assign(train_times , train_trunk + train_times,\
                      name="train_times_add")            
        merged_summary = tfs.merge_all() 
       
    return {"graph":graph,"merged_summary":merged_summary}

def read_config_file(trainning_task_file)    :
    """
    read_config_file函数用来从配置文件读取参数
    返回一个字典，主要包括如下参数：
       参数名          意义                   对应配置文件路径
    1. n               输入向量维度（特征数） network/input_nodes
    2. K               输出向量维度（判决数） network/output_nodes
    3. dlambda         正则化参数（0,001）    network/lambda
    4. ns_array        各个隐层的规模，list   network/hidden_layer_size
    5. file_deal_times 本训练文件训练次数     network/file_deal_times
    6. trunk           参与训练样本数         network/trunk
    7. train_step      参与训练窗口下移样本数 network/train_step
    8. iterate_times   参与训练窗口迭代次数   network/iterate_times
    """
    #reading config file
    config = configparser.ConfigParser()
    config.read(trainning_task_file)
    n       = int(config['network']['input_nodes'])     # input vector size
    K       = int(config['network']['output_nodes'])     # output vector size
    lamda           = float(config['network']['lambda'])
    #hidden layer size, string split with ",",like ”16,16,13“ 
    hidden_layer_size = config['network']['hidden_layer_size'] 
    #split each layer size
    reobj = re.compile('[\s,\"]')
    ls_array = reobj.split(hidden_layer_size)
    #remove null strings
    ls_array = [item for item in filter(lambda x:x != '', ls_array)] 
    #get hidden layer size
    hidden_layer_size =  len(ls_array)
    
    #convert 
    ns_array = []
    for idx in range(0,hidden_layer_size)    :
        ns_array.append(int(ls_array[idx]))

    file_deal_times = int(config['performance']['file_deal_times'])
    trunk           = int(config['performance']['trunk'])
    train_step      = int(config['performance']['train_step'])
    iterate_times   = int(config['performance']['iterate_times'])

    return {"n":n,"K":K,"dlambda":lamda,"ns_array":ns_array,\
            "file_deal_times":file_deal_times,"trunk":trunk,\
            "train_step":train_step,"iterate_times":iterate_times}

def init_graph (graph,sess):
    """
    init_graph函数用来在一个会话中初始化一个图
    图是一个结构，会话是tf用来运行任务的上下文。
    """
    with graph.as_default():
        #save graph to Disk
        init = tf.global_variables_initializer()        
        sess.run(init)          # Very important

def train_model(trainning_task_file,trainning_input_file,model_path,first,\
                summary_path = "./network"):
    """
    #train_model函数用来训练模型
    @para trainning_task_file  cfg文件名，参见函数 read_config_file
    @para trainning_input_file 训练样本文件名。每行一组样本，包含逗号\
                               分割的特征向量,训练向量（标签或者理论输出）
    @para model_path           存储训练结果的路径
    @para first                是否是首次训练。=0为增量学习，model_path必须存在。
    @return 返回学习曲线
    """
    #reading config file
    cfObj = read_config_file(trainning_task_file)
    n               = cfObj["n"]
    K               = cfObj["K"]
    lamda           = cfObj["dlambda"]
    ns_array        = cfObj["ns_array"]
    
    #create graph
    net_wk = create_graph(n,K,lamda,ns_array)
    graph = net_wk["graph"]
    merged_summary = net_wk["merged_summary"]
    with graph.as_default():
        #save graph to Disk
        saver = tf.train.Saver()
        loss = graph.get_tensor_by_name("loss/loss:0")
        train = graph.get_operation_by_name("train/train")
        
    ### create tensorflow structure end ###
    sess = tf.Session(graph=graph)
    if (first != 0):
        init_graph(graph,sess)
    else:
        check_point_path = model_path # 保存好模型的文件路径
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=check_point_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
            
    #writer = tf.summary.FileWriter("./netdemo/")
    #writer.add_graph(sess.graph)
    #writer.close()
    file_deal_times = cfObj["file_deal_times"]
    trunk           = cfObj["trunk"]
    train_step      = cfObj["train_step"]
    iterate_times   = cfObj["iterate_times"]
    #trainning
    x_data = np.zeros([trunk,n]).astype(np.float32)
    #read n features and K outputs
    y_data = np.zeros([trunk,K]).astype(np.float32)
    total_red = 0
    
    x_plot = []
    y_plot = []
    
    ainput = graph.get_tensor_by_name("network/input:0")
    tout = graph.get_tensor_by_name("loss/tr_out:0")
    
    #训练情况
    train_trunk = graph.get_tensor_by_name("train/train_trunk:0")
    train_times_add = graph.get_tensor_by_name("train/train_times_add:0")
           
    
    writer = tf.summary.FileWriter(summary_path)
    writer.add_graph(graph) 
    
    reobj = re.compile('[\s,\"]')
    
    trk = np.ones([1,1]) * iterate_times
    
    for rc in range(file_deal_times):
        with open(trainning_input_file, 'rt') as ftr:
            while 1:
                lines = ftr.readlines()
                if not lines:
                    #reach end of file, run trainning for tail items if \
                    #there is some.
                    if (total_red>0):
                        for step in range(iterate_times):
                            sess.run(train,\
                                     feed_dict={ainput:x_data[0:min(total_red,\
                                    trunk)+1],tout:y_data[0:min(total_red,\
                                         trunk)+1]})
                    break
                line_count = len(lines)
                for lct in range(line_count):
                    x_arr = reobj.split(lines[lct])
                    #remove null strings
                    x_arr = [item for item in filter(lambda x:x != '', x_arr)] 
                    for idx in range(n)    :
                        x_data[total_red % trunk,idx] = float(x_arr[idx])
                    for idx in range(K)    :    
                        y_data[total_red % trunk,idx] = float(x_arr[idx+n])           
                    total_red = total_red + 1
                    #the trainning set run trainning
                    if (total_red % train_step == 0):
                        #trainning
                        for step in range(iterate_times):
                            sess.run(train,feed_dict=\
                                     {ainput:x_data[0:min(total_red,trunk)+1],\
                                        tout:y_data[0:min(total_red,trunk)+1]})
                        #可视化
                        t_count = sess.run(train_times_add,\
                                           feed_dict={train_trunk:trk})
                        s = sess.run(merged_summary,\
                                     feed_dict={ainput:x_data[0:min(total_red,\
                                    trunk)+1],tout:y_data[0:min(total_red,\
                                     trunk)+1]})
                        writer.add_summary(s,t_count)
                        #print loss
                        lss = sess.run(loss,feed_dict=\
                                       {ainput:x_data[0:min(total_red,\
                                        trunk)+1],tout:y_data[0:min(total_red,\
                                         trunk)+1]})
                        print(rc,total_red,t_count,lss)
                        x_plot.append(total_red/1000)
                        y_plot.append(lss)
                        saver.save(sess,model_path+'/model.ckpt')
            
    #saving
    # 保存，这次就可以成功了
    saver.save(sess,model_path+'/model.ckpt')
    #mp.plot(x_plot,y_plot,'b')
    return {"x_plot":x_plot,"y_plot":y_plot,"graph":graph}
     
def test_model(trainning_task_file,testing_file,model_path):
    """
    test_model函数用来测试模型的收敛
    @para trainning_task_file  cfg文件名，参见函数 read_config_file
    @para testing_file         测试样本文件名。每行一组样本，包含逗号分割的特征向\
                                量,训练向量（标签或者理论输出）
    @para model_path           存储训练结果的路径
    @return 返回学习曲线
    """
    #reading config file
    cfObj = read_config_file(trainning_task_file)
    n               = cfObj["n"]
    K               = cfObj["K"]
    lamda           = cfObj["dlambda"]
    ns_array        = cfObj["ns_array"]
    
    #create graph
    net_wk = create_graph(n,K,lamda,ns_array)
    graph = net_wk["graph"]
    #merged_summary = net_wk["merged_summary"]
    with graph.as_default():
        #save graph to Disk
        saver = tf.train.Saver()
        loss = graph.get_tensor_by_name("loss/loss:0")
        
    ### create tensorflow structure end ###
    sess = tf.Session(graph=graph)
    check_point_path = model_path # 保存好模型的文件路径
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir=check_point_path)
    saver.restore(sess,ckpt.model_checkpoint_path)
    
    #--------------------------------------------------------------
    trunk           = cfObj["trunk"]
    train_step      = cfObj["train_step"]
    
    ainput = sess.graph.get_tensor_by_name("network/input:0")
    tout = graph.get_tensor_by_name("loss/tr_out:0")
    
    print ("Testing...")
    #testing
    x_test = np.zeros([trunk,n]).astype(np.float32)
    #read n features and K outputs
    y_test = np.zeros([trunk,K]).astype(np.float32)
    x_plot = []
    y_plot = []

    total_red = 0
    reobj = re.compile('[\s,\"]')
    with open(testing_file, 'rt') as testfile:
        while 1:
            lines = testfile.readlines()
            if not lines:
                break
            line_count = len(lines)
            for lct in range(line_count):
                x_arr = reobj.split(lines[lct])
                #remove null strings
                x_arr = [item for item in filter(lambda x:x != '', x_arr)] 
                for idx in range(n)    :
                    x_test[total_red % trunk,idx] = float(x_arr[idx])
                for idx in range(K)    :    
                    y_test[total_red % trunk,idx] = float(x_arr[idx+n])           
                total_red = total_red + 1
                #the trainning set run trainning
                if (total_red % train_step == 0):
                    #print loss
                    lss = sess.run(loss,feed_dict=\
                               {ainput:x_test[0:min(total_red,trunk)+1],\
                                  tout:y_test[0:min(total_red,trunk)+1]})
                    print(total_red,lss)
                    x_plot.append(total_red/1000)
                    y_plot.append(lss)

    
    return {"x_plot":x_plot,"y_plot":y_plot,"graph":graph}

def run_model(trainning_task_file,input_file,model_path,output_file):
    """
    run_model函数用来应用模型
    @para trainning_task_file  cfg文件名，参见函数 read_config_file
    @para input_file           输入文件名。每行一组样本，包含逗号分割的特征向量
    @para model_path           存储训练结果的路径
    @para output_file          输出文件名。每行一组样本，
                                包含逗号分割的[特征向量],[输出结果]
    @return 返回字典：样本(x_test)、结果(result)
    """
    #reading config file
    cfObj = read_config_file(trainning_task_file)
    n               = cfObj["n"]
    K               = cfObj["K"]
    lamda           = cfObj["dlambda"]
    ns_array        = cfObj["ns_array"]
    #--------------------------------------------------------------
    #create graph

    net_wk = create_graph(n,K,lamda,ns_array)
    graph = net_wk["graph"]
    #merged_summary = net_wk["merged_summary"]
    with graph.as_default():
        #save graph to Disk
        saver = tf.train.Saver()
        
    ### create tensorflow structure end ###
    sess = tf.Session(graph=graph)
    check_point_path = model_path # 保存好模型的文件路径
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir=check_point_path)
    saver.restore(sess,ckpt.model_checkpoint_path)
    
    ainput = sess.graph.get_tensor_by_name("network/input:0")
    outv = graph.get_tensor_by_name("network/output:0")
    reobj = re.compile('[\s,\"]')
    #--------------------------------------------------------------
    print ("Running...")
    with open(input_file, 'rt') as testfile:
        with open(output_file, 'wt') as resultfile:    
            while 1:
                lines = testfile.readlines()
                if not lines:
                    break
                line_count = len(lines)
                x_test = np.zeros([line_count,n]).astype(np.float32)
                for lct in range(line_count):
                    x_arr = reobj.split(lines[lct])
                    #remove null strings
                    x_arr = [item for item in filter(lambda x:x != '', x_arr)] 
                    for idx in range(n)    :
                        x_test[lct,idx] = float(x_arr[idx])
                #the trainning set run trainning
                result = sess.run(outv,feed_dict={ainput:x_test})
                for idx in range(line_count):
                    print(x_test[idx].tolist(),result[idx].tolist(), file = resultfile)
    return {"x_test":x_test,"result":result,"graph":graph}
   
def dump_all(trainning_task_file,model_path):
    """
    dump_all 是用于提取所有矩阵信息的函数。
    """
    #reading config file    
    cfObj = read_config_file(trainning_task_file)
    n               = cfObj["n"]
    K               = cfObj["K"]
    lamda           = cfObj["dlambda"]
    ns_array        = cfObj["ns_array"]
    net_wk = create_graph(n,K,lamda,ns_array)
    graph = net_wk["graph"]
    #merged_summary = net_wk["merged_summary"]
    with graph.as_default():
        #save graph to Disk
        saver = tf.train.Saver()
        
    ### create tensorflow structure end ###
    sess = tf.Session(graph=graph)
    check_point_path = model_path # 保存好模型的文件路径
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir=check_point_path)
    saver.restore(sess,ckpt.model_checkpoint_path)

    #Output is the last layer, append to last
    ns_array        = cfObj["ns_array"]
    ns_array.append(K)
    hidden_layer_size = len(ns_array)
    #--------------------------------------------------------------
    W = []
    b = []
    z = []
    punish = tf.constant(0.0, name='regular')
    for idx in range(0,hidden_layer_size)    :
        sess_v = sess.graph.get_tensor_by_name("network/"+'W'+str(idx+1)+":0")
        W.append(sess.run(sess_v))
        sess_v = sess.graph.get_tensor_by_name("network/"+'b'+str(idx+1)+":0")
        b.append(sess.run(sess_v))
    sess_v = sess.graph.get_tensor_by_name("network/regular:0")
    punish = sess.run(sess_v)
    
    cfObj["W"] = W
    cfObj["b"] = b
    cfObj["z"] = z
    cfObj["punish"] = punish
    cfObj["relation"] = "a' = f (a * W + b)"
    sess_v = sess.graph.get_tensor_by_name("train/train_times:0")
    train_times = sess.run(sess_v)
    cfObj["train_times"] = train_times
    return cfObj