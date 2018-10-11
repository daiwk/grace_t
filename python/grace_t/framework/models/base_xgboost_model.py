#!/usr/bin/env python
# -*- coding: utf8 -*-
 
import sys
import os
import ConfigParser
import base_model

base_path = os.path.dirname(os.path.abspath(__file__)) + "/../../"
os.sys.path.append(base_path)
DEFAULT_LOG_FILENAME = base_path + "/log/base_xgboost_model"
DEFAULT_MODEL_PATH = base_path + "/model/"

import logging

import numpy

from numpy import loadtxt

import sklearn
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot

class BaseXGBoostModel(base_model.BaseModel):
    '''
    base xgboost model based on xgboost's model
    '''
    def __init__(self, **kargs):
        '''
        init
        '''
        import framework.tools.log as log
        self.kargs = kargs
        self.dic_params = {}
        log_filename = self.kargs["basic_params"]["log_filename"]
        model_path = self.kargs["basic_params"]["model_path"]
        self.load_data_func = self.kargs["load_data"]["method"]
        self.create_model_func = self.kargs["create_model"]["method"]
        loger = log.init_log(log_filename)
        (self.dataset, self.X, self.Y, self.X_evaluation, self.Y_evaluation) = self.load_data_func(**self.kargs["load_data"]["params"])
        self.model_path = model_path
        dic_params = {}

    def init_model(self):
        '''
        init model
        '''
        self.model = self.create_model_func(**self.kargs["create_model"]["params"])
    
    def train_model(self, ):
        '''
        train model
        '''
        X = self.X
        Y = self.Y
        X_evaluation = self.X_evaluation
        Y_evaluation = self.Y_evaluation
        train_params = {"eval_metric": "error", "verbose": True} # "early_stopping_rounds": 100, 
        self.dic_params.update(train_params)
        self.model.fit(X, Y, **self.dic_params) # Evaluate the model
        print "feature importances:", self.model.feature_importances_
        plot_importance(self.model)
        pyplot.show()
        
    
    def process(self):
        '''
        process
        '''
        #self.init_callbacks()
        self.init_model()
        self.train_model()

