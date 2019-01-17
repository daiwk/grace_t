# Copyright 2018 The grace_t Authors. All Rights Reserved.
#  
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#  
# http://www.apache.org/licenses/LICENSE-2.0
#  
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import matplotlib
matplotlib.use('Agg')
import xgboost as xgb
import matplotlib.pyplot as plt

model_file = "xgb_dict_file"
bst = xgb.Booster({'nthread': 4}) 
bst.load_model(model_file)

fig,ax = plt.subplots()
fig.set_size_inches(60,30)
xgb.plot_tree(bst,ax = ax)
fig.savefig('xgb_tree.jpg')
fig,ax = plt.subplots()
fig.set_size_inches(60,30)
xgb.plot_importance(bst,ax = ax)
fig.savefig('xgb_tree_importance.jpg')
