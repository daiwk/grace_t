#!/usr/bin/env python
# -*- coding: utf8 -*-
 
import sys
import os
import ConfigParser
import base_model

base_path = os.path.dirname(os.path.abspath(__file__)) + "/../../"
os.sys.path.append(base_path)
DEFAULT_LOG_FILENAME = base_path + "/log/base_keras_model"
DEFAULT_MODEL_PATH = base_path + "/model/"

import logging

import numpy

from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import Dropout

from keras.wrappers.scikit_learn import KerasClassifier

from keras.optimizers import SGD

from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback

import sklearn

class LossHistory(Callback):
    '''
    loss history
    '''
    def on_train_begin(self, logs={}):
        '''
        on_train_begin
        '''
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        '''
        on_batch_end
        '''
        self.losses.append(logs.get('loss'))


def create_model_demo():
    '''
    create model
    '''
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


class BaseKerasSklearnModel(base_model.BaseModel):
    '''
    base keras model based on keras's model(without sklearn)
    '''
##    def __init__(self, data_file, delimiter, lst_x_keys, lst_y_keys, log_filename=DEFAULT_LOG_FILENAME, model_path=DEFAULT_MODEL_PATH, create_model_func=create_model_demo):
##        '''
##        init
##        '''
##        import framework.tools.log as log
##        loger = log.init_log(log_filename)
##        self.load_data(data_file, delimiter, lst_x_keys, lst_y_keys)
##        self.model_path = model_path
##        self.create_model_func=create_model_func

    def __init__(self, **kargs):
        '''
        init
        '''
        import framework.tools.log as log
        self.kargs = kargs
        log_filename = self.kargs["basic_params"]["log_filename"]
        model_path = self.kargs["basic_params"]["model_path"]
        self.load_data_func = self.kargs["load_data"]["method"]
        self.create_model_func = self.kargs["create_model"]["method"]
        loger = log.init_log(log_filename)
        (self.dataset, self.X, self.Y, self.X_evaluation, self.Y_evaluation) = self.load_data_func(**self.kargs["load_data"]["params"])
        self.model_path = model_path
        self.dic_params = {}
 

    def load_data(self, data_file, delimiter, lst_x_keys, lst_y_keys):
        '''
        load data
        '''
        # Load the dataset
        self.dataset = numpy.loadtxt(data_file, delimiter=",") 
        self.X = self.dataset[:, lst_x_keys] 
        self.Y = self.dataset[:, lst_y_keys]
    
    def init_callbacks(self):
        '''
        init all callbacks
        '''
        os.system("mkdir -p %s" % (self.model_path))
        checkpoint_callback = ModelCheckpoint(self.model_path + '/weights.{epoch:02d}-{acc:.2f}.hdf5', \
                monitor='acc', save_best_only=False)
        history_callback = LossHistory()
        callbacks_list = [checkpoint_callback, history_callback]
        self.dic_params["callbacks"] = callbacks_list

    def init_model(self):
        '''
        init model
        '''
        train_params = {"nb_epoch": 10, "batch_size": 10}
        self.dic_params.update(train_params)
        self.model = KerasClassifier(build_fn=self.create_model_func, **self.kargs["create_model"]["params"])
#        self.model = KerasClassifier(build_fn=self.create_model_func)
        self.model.set_params(**self.dic_params)
    
    def train_model(self):
        '''
        train model
        '''
        X = self.X
        Y = self.Y
        X_evaluation = self.X_evaluation
        Y_evaluation = self.Y_evaluation
        seed = 7
        numpy.random.seed(seed) # Load the dataset
        
        history = self.model.fit(X, Y)
        scores = self.model.score(X, Y)
#history_callback = self.dic_params["callbacks"][1]
#        print dir(history_callback)
#        logging.info(str(history_callback.losses))
        logging.info("final : %.2f%%" % (scores * 100))
        logging.info(str(history.history))
    
    def process(self):
        '''
        process
        '''
        self.init_callbacks()
        self.init_model()
        self.train_model()

