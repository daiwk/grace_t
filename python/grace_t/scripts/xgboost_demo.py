#!/usr/bin/env python
# -*- coding: utf8 -*-
 
import os
import ConfigParser
import logging

import numpy

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from common_import import BaseXGBoostModel as BaseXGBoostModel

base_path = os.path.dirname(os.path.abspath(__file__)) + "/../"
os.sys.path.append(base_path)
config_path = base_path + '/conf/'
data_path = base_path + '/data/'


def my_xgboost_load_data(data_file, delimiter, lst_x_keys, lst_y_keys):
    '''
    my_xgboost_load_data
    '''
    dataset = numpy.loadtxt(data_file, delimiter=delimiter)
    # split data into X and y
    X = dataset[:, lst_x_keys]
    Y = dataset[:,lst_y_keys]
    # split data into train and test sets
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    return (dataset, X_train, y_train, X_test, y_test)


def my_xgboost_create_model():
    '''
    create a xgboost model
    '''
    model = XGBClassifier()

    return model
 

def train_xgboost_demo():
    '''
    train_xgboost_demo
    '''
    ### base
    data_file = data_path + 'pima-indians-diabetes.csv'
    lst_x_keys = list(xrange(0, 8))
    lst_y_keys = 8
    delimiter = ','

    config_file = config_path + 'demo_base_xgboost.conf'
    conf = ConfigParser.ConfigParser()
    conf.read(config_file)
    demo_log_file = base_path + "log/" + conf.get("log", "log_name")
    demo_model_path = base_path + "model/" + conf.get("model", "model_path")
    
    dict_params = {}
    dict_params["load_data"] = {}
    dict_params["create_model"] = {}
    dict_params["basic_params"] = {}

    dict_load_data = dict_params["load_data"]
    dict_load_data["method"] = my_xgboost_load_data
    dict_load_data["params"] = {}
    dict_load_data_params = dict_load_data["params"]
    dict_load_data_params["data_file"] = data_file
    dict_load_data_params["delimiter"] = delimiter
    dict_load_data_params["lst_x_keys"] = lst_x_keys
    dict_load_data_params["lst_y_keys"] = lst_y_keys
   
    dict_create_model = dict_params["create_model"]
    dict_create_model["method"] = my_xgboost_create_model
    dict_create_model["params"] = {}
    dict_create_model_params = dict_create_model["params"]

    dict_basic_params = dict_params["basic_params"]
    dict_basic_params["log_filename"] = demo_log_file
    dict_basic_params["model_path"] = demo_model_path

    base_inst = BaseXGBoostModel(**dict_params)
    base_inst.process()



if "__main__" == __name__:

    train_xgboost_demo()

    exit(0)
