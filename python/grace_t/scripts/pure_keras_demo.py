#!/usr/bin/env python
# -*- coding: gbk -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: grace_t/python/grace_t/scripts/pure_keras_demo.py
Author: wkdai(wkdai@baidu.com)
Date: 2019/02/10 13:44:31
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Convolution1D, GlobalAveragePooling1D, MaxPooling1D, GlobalMaxPooling1D

model = Sequential()
model.add(Convolution1D(64, 3, input_shape=(10, 32))) #output 8,64
#model.add(MaxPooling1D(8,input_shape=(8, 32))) #output 1,64
model.add(GlobalMaxPooling1D(input_shape=(8, 32))) #output 1,64

print(model.summary())
