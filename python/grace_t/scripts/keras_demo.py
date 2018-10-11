#!/usr/bin/env python
# -*- coding: utf8 -*-
 
import os
import ConfigParser
import logging

import numpy

from keras.utils import np_utils
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import Dropout

from keras.optimizers import SGD

# cnn
from keras.datasets import mnist
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten


from common_import import BaseKerasSklearnModel as BaseKerasSklearnModel
from common_import import BaseKerasModel as BaseKerasModel

base_path = os.path.dirname(os.path.abspath(__file__)) + "/../"
os.sys.path.append(base_path)
config_path = base_path + '/conf/'
data_path = base_path + '/data/'


def my_cnn_load_data():
    '''
    my_cnn_load_data
    '''
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28) 
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    classes = set()
    for i in y_train:
        classes.add(i)
    y_train = np_utils.to_categorical(y_train)
    
    y_test = np_utils.to_categorical(y_test)

    dataset = (X_train, y_train)

    return (dataset, X_train, y_train, X_test, y_test)


def my_cnn_create_model(num_classes):
    '''
    create a cnn model
    '''
    # Define and Compile 
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    sgd = SGD(lr=0.1, momentum=0.9, decay=0.0001, nesterov=True) # learning rate schedule
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) # Fit the model
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Fit the model
    return model
 

def my_load_data(data_file, delimiter, lst_x_keys, lst_y_keys):
    '''
    my load data
    '''
    # Load the dataset
    dataset = numpy.loadtxt(data_file, delimiter=delimiter) 
    X = dataset[:, lst_x_keys] 
    Y = dataset[:, lst_y_keys]
    ## mock
    X_test = X
    Y_test = Y 
    return (dataset, X, Y, X_test, Y_test)
 

def my_create_model():
    '''
    my create model
    '''
    print "using self defined create_model!!"
    # Define and Compile 
    model = Sequential()
    model.add(Dense(12, input_dim=8, init='uniform', activation='relu')) 
    model.add(Dense(8, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    model.add(Dropout(0.2))

    sgd = SGD(lr=0.1, momentum=0.9, decay=0.0001, nesterov=True) # learning rate schedule
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy']) # Fit the model
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Fit the model
    return model
 

def train_keras_demo():
    '''
    train_keras_demo
    '''
    data_file = data_path + 'pima-indians-diabetes.csv'
    lst_x_keys = list(xrange(0, 8))
    lst_y_keys = 8
    delimiter = ','
    ### base
    config_file = config_path + 'demo_base.conf'
    conf = ConfigParser.ConfigParser()
    conf.read(config_file)
    demo_log_file = base_path + "log/" + conf.get("log", "log_name")
    demo_model_path = base_path + "model/" + conf.get("model", "model_path")

#base_inst = BaseKerasModel(data_file, delimiter, lst_x_keys, lst_y_keys, \
#            log_filename=demo_log_file, model_path=demo_model_path, create_model_func=create_model)
    dict_params = {}
    dict_params["load_data"] = {}
    dict_params["create_model"] = {}
    dict_params["basic_params"] = {}

    dict_load_data = dict_params["load_data"]
    dict_load_data["method"] = my_load_data
    dict_load_data["params"] = {}
    dict_load_data_params = dict_load_data["params"]
    dict_load_data_params["data_file"] = data_file
    dict_load_data_params["delimiter"] = delimiter
    dict_load_data_params["lst_x_keys"] = lst_x_keys
    dict_load_data_params["lst_y_keys"] = lst_y_keys
   
    dict_create_model = dict_params["create_model"]
    dict_create_model["method"] = my_create_model
    dict_create_model["params"] = {}

    dict_basic_params = dict_params["basic_params"]
    dict_basic_params["log_filename"] = demo_log_file
    dict_basic_params["model_path"] = demo_model_path

    base_inst = BaseKerasModel(**dict_params)
    base_inst.process()
    return 0


def train_keras_sklearn_demo():
    '''
    train_keras_sklearn_demo
    '''
    ## sklearn
    data_file = data_path + 'pima-indians-diabetes.csv'
    lst_x_keys = list(xrange(0, 8))
    lst_y_keys = 8
    delimiter = ','

    config_file = config_path + 'demo_base_sklearn.conf'
    conf = ConfigParser.ConfigParser()
    conf.read(config_file)
    demo_log_file = base_path + "log/" + conf.get("log", "log_name")
    demo_model_path = base_path + "model/" + conf.get("model", "model_path")

    dict_params = {}
    dict_params["load_data"] = {}
    dict_params["create_model"] = {}
    dict_params["basic_params"] = {}

    dict_load_data = dict_params["load_data"]
    dict_load_data["method"] = my_load_data
    dict_load_data["params"] = {}
    dict_load_data_params = dict_load_data["params"]
    dict_load_data_params["data_file"] = data_file
    dict_load_data_params["delimiter"] = delimiter
    dict_load_data_params["lst_x_keys"] = lst_x_keys
    dict_load_data_params["lst_y_keys"] = lst_y_keys
   
    dict_create_model = dict_params["create_model"]
    dict_create_model["method"] = my_create_model
    dict_create_model["params"] = {}

    dict_basic_params = dict_params["basic_params"]
    dict_basic_params["log_filename"] = demo_log_file
    dict_basic_params["model_path"] = demo_model_path


##    base_sklearn_inst = BaseKerasSklearnModel(data_file, delimiter, lst_x_keys, lst_y_keys, \
##            log_filename=demo_log_file, model_path=demo_model_path, \
##            create_model_func=create_model)
    base_sklearn_inst = BaseKerasSklearnModel(**dict_params)

    base_sklearn_inst.process()
    return 0


def train_keras_cnn_demo():
    '''
    train_keras_cnn_demo
    '''
    ### base
    config_file = config_path + 'demo_base.conf'
    conf = ConfigParser.ConfigParser()
    conf.read(config_file)
    demo_log_file = base_path + "log/" + conf.get("log", "log_name")
    demo_model_path = base_path + "model/" + conf.get("model", "model_path")
    
    #num_classes = len(classes)
    num_classes = 10
    
    dict_params = {}
    dict_params["load_data"] = {}
    dict_params["create_model"] = {}
    dict_params["basic_params"] = {}

    dict_load_data = dict_params["load_data"]
    dict_load_data["method"] = my_cnn_load_data
    dict_load_data["params"] = {}
    dict_load_data_params = dict_load_data["params"]
#    dict_load_data_params["delimiter"] = delimiter
#    dict_load_data_params["lst_x_keys"] = lst_x_keys
#    dict_load_data_params["lst_y_keys"] = lst_y_keys
   
    dict_create_model = dict_params["create_model"]
    dict_create_model["method"] = my_cnn_create_model
    dict_create_model["params"] = {}
    dict_create_model_params = dict_create_model["params"]
    dict_create_model_params["num_classes"] = num_classes

    dict_basic_params = dict_params["basic_params"]
    dict_basic_params["log_filename"] = demo_log_file
    dict_basic_params["model_path"] = demo_model_path

    base_inst = BaseKerasModel(**dict_params)
    base_inst.process()


if "__main__" == __name__:

    train_keras_demo()
    train_keras_sklearn_demo()
    train_keras_cnn_demo()

    exit(0)
