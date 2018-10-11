#!/usr/bin/env python
# -*- coding: utf8 -*-
# author: flyzzaway

##export PATH=/opt/compiler/gcc-4.8.2/bin/:$PATH
##import theano
##theano.config.openmp = True
##OMP_NUM_THREADS=20 python xxx.py
##就可以跑多核cpu了。。另外 装个jumbo install htop，可以看到每个核的占用情况。。

import numpy as np
import pandas as pd
import gc
from keras.models import Sequential
from keras.layers import Dense,Dropout,Embedding,Concatenate,Activation,Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from ml_metrics import mapk
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
def get_max():
    '''
    get_max
    '''
    
    max_dict = {}
    max_dict['max_displayid'] = 172668
    max_dict['max_adid'] = 353356
    max_dict['max_platform'] = 4
    max_dict['max_hour'] = 6
    max_dict['max_weekday'] = 1
    max_dict['max_uid'] = 166049
    max_dict['max_documentid'] = 1802181
    max_dict['max_campaignid'] = 29471
    max_dict['max_advertiserid'] = 4036
    max_dict['max_sourceidx'] = 14403
    max_dict['max_categoryid'] = 95
    max_dict['max_entityid'] = 1326010
    max_dict['max_topicid'] = 301
    max_dict['max_doctrfids'] = 2998870 
    max_dict['max_sourceidy'] = 14404
    max_dict['max_docids'] = 2997096
    return max_dict


def sub_input_model(max_dict,embedding_size,max_name,input_shape_dim1,name):
    '''
    sub_input_model
    '''
    print "sub_input_model"
    sub_model = Sequential()
    sub_model.add(Embedding(input_dim = max_dict[max_name], output_dim=embedding_size,input_shape=(input_shape_dim1,),name = name)) #input_shape=(1,)
    sub_model.add(Flatten(name = 'Flatten' + name))
    sub_model.add(Dense(name = 'Dense' + name))
    return sub_model

def multi_input_model(model,max_dict):
    '''
    multi_input_model
    '''
    print "multi_input_model"
 
    model_displayid = sub_input_model(max_dict,embedding_size = 50,max_name = 'max_displayid',input_shape_dim1 = 1,name='displayid')
    
    model_adid = sub_input_model(max_dict,embedding_size = 50,max_name = 'max_adid',input_shape_dim1 = 1,name='adid')
    
    model_platform = sub_input_model(max_dict,embedding_size = 50,max_name = 'max_platform',input_shape_dim1 = 1,name = 'platform')
    
    model_hour = sub_input_model(max_dict,embedding_size = 50,max_name = 'max_hour',input_shape_dim1 = 1,name = 'hour')
    
    model_weekday = sub_input_model(max_dict,embedding_size = 50,max_name = 'max_weekday',input_shape_dim1 = 1 ,name = 'weekday')
    
    model_uid = sub_input_model(max_dict,embedding_size = 50,max_name = 'max_uid',input_shape_dim1 = 1, name = 'uid')
    
    model_documentid = sub_input_model(max_dict,embedding_size = 50,max_name = 'max_documentid',input_shape_dim1 = 1,name = 'documentid')
    
    model_campaignid = sub_input_model(max_dict,embedding_size = 50,max_name = 'max_campaignid',input_shape_dim1 = 1,name = 'campaignid')
    
    model_advertiserid = sub_input_model(max_dict,embedding_size = 50,max_name = 'max_advertiserid',input_shape_dim1 = 1, name = 'advertiserid')
 
    model_sourceidx = sub_input_model(max_dict,embedding_size = 50,max_name = 'max_sourceidx',input_shape_dim1 = 1,name = 'sourceid')
   
    model_categoryid = sub_input_model(max_dict,embedding_size = 50,max_name = 'max_categoryid',input_shape_dim1 = 2,name = 'categoryid')
    
    model_entityid = sub_input_model(max_dict,embedding_size = 50,max_name = 'max_entityid',input_shape_dim1 = 10,name = 'entityid')
    
    model_topicid = sub_input_model(max_dict,embedding_size = 50,max_name = 'max_topicid',input_shape_dim1 = 39,name = 'topicid') 
    
    model_uidview_doc = sub_input_model(max_dict,embedding_size = 50,max_name = 'max_doctrfids',input_shape_dim1 = 306,name = 'uidview_doc')
    
    model_uidview_source = sub_input_model(max_dict,embedding_size = 50,max_name = 'max_sourceidy',input_shape_dim1 = 160,name = 'uidview_source')
    
    model_uidview_onehour_doc = sub_input_model(max_dict,embedding_size = 50,max_name = 'max_docids',input_shape_dim1 = 123,name = 'uidview_onehour_doc')
    
    model.add(Concatenate([model_displayid, model_adid, model_platform,model_hour,model_weekday,model_uid,model_documentid,model_campaignid,\
                     model_advertiserid,model_sourceidx,model_categoryid,model_entityid,model_topicid,model_uidview_doc,\
                      model_uidview_source,model_uidview_onehour_doc], mode='concat', concat_axis=1))
    
    print('the model\'s input shape ', model.input_shape)
    print ('the mode\'s output shape ', model.output_shape)
    
    model.add(Dense(30,activation = 'relu',name = 'Dense_1'))
    print model.output_shape
    
    model.add(Dense(1, activation='sigmoid',name = 'Dense_2'))
    print('the final model\'s shape', model.output_shape)
    
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy', 'mae'])

    plot_model(model,to_file='model.png', show_shapes=True)


if __name__ == "__main__":
# rand 300 training examples
#    x_train_displayid = np.random.randint(30000,size=(300,1))
#    x_train_adid = np.random.randint(220000,size=(300,1))
#    x_train_entityids = np.random.randint(4000000,size=(300,10))
#    y_train = np.random.randint(1,size=(300,1+1+10,1))
#    print y_train.shape
    
    max_dict = get_max() 
    model = Sequential()
    multi_input_model(model,max_dict)


   
#    model.fit([x_train_displayid, x_train_adid, x_train_entityids], y_train, batch_size=16, epochs=10)
#    score = model.evaluate([x_test_1, x_test_2], y_test, batch_size=16)
#    print score
