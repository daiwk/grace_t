# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""tt"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

# pylint: disable=wrong-import-order
from absl import app as absl_app
from absl import flags
from six.moves import urllib
import tensorflow as tf
# pylint: enable=wrong-import-order

from official.utils.flags import core as flags_core

TRAINING_FILE = 'tt.data'
EVAL_FILE = 'tt.test'
SUBMISSION_FILE = 'tt.submission'

_CSV_COLUMNS = [
        "ad_stuff_size_show", "adid", "ad_account_id", "ad_product_id",
        "age", "education", "consumption_ability", 
        "bid", 
        "show"]

_CSV_COLUMN_DEFAULTS = [
        [0], [''], [''], [''], 
        [''], [''], [''], 
        [0.0], 
        [0.0]]

_HASH_BUCKET_SIZE = 1000
ADID_HASH_BUCKET_SIZE = 740000

idx = 0
with open("./data/tt_data/tt.data", 'rb') as fin:
    for line in fin:
        idx += 1
train_num = idx

idx = 0
with open("./data/tt_data/tt.test", 'rb') as fin:
    for line in fin:
        idx += 1
val_num = idx

_NUM_EXAMPLES = {
    'train': train_num,
    'validation': val_num,
}

def build_model_columns():
  """Builds a set of wide and deep feature columns."""
  # Continuous variable columns
  bid = tf.feature_column.numeric_column('bid')
  ad_stuff_size_show = tf.feature_column.numeric_column('ad_stuff_size_show')
  #education = tf.feature_column.categorical_column_with_vocabulary_list(
  #    'education', [
  #        'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
  #        'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
  #        '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

  consumption_ability = tf.feature_column.categorical_column_with_hash_bucket(
      'consumption_ability', hash_bucket_size=2)

  age = tf.feature_column.categorical_column_with_hash_bucket(
      'age', hash_bucket_size=_HASH_BUCKET_SIZE)

  education = tf.feature_column.categorical_column_with_hash_bucket(
      'education', hash_bucket_size=6)

  adid = tf.feature_column.categorical_column_with_hash_bucket(
      'adid', hash_bucket_size=ADID_HASH_BUCKET_SIZE)

  ad_product_id = tf.feature_column.categorical_column_with_hash_bucket(
      'ad_product_id', hash_bucket_size=34000)

  ad_account_id = tf.feature_column.categorical_column_with_hash_bucket(
      'ad_account_id', hash_bucket_size=30000)

  #age_buckets = tf.feature_column.bucketized_column(
  #    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

  # Wide columns and deep columns.
  base_columns = [
      adid, ad_account_id, ad_product_id, #education, consumption_ability, age
  ]

  crossed_columns = [
      tf.feature_column.crossed_column(
          ['ad_account_id', 'ad_product_id'], hash_bucket_size=ADID_HASH_BUCKET_SIZE),
##      tf.feature_column.crossed_column(
##          ['education', 'consumption_ability'], hash_bucket_size=_HASH_BUCKET_SIZE),
##      tf.feature_column.crossed_column(
##          ['age', 'consumption_ability'],
##          hash_bucket_size=_HASH_BUCKET_SIZE),
  ]

  wide_columns = base_columns + crossed_columns

  deep_columns = [
      bid, 
      ad_stuff_size_show,
      tf.feature_column.indicator_column(education),
      tf.feature_column.indicator_column(consumption_ability),
      tf.feature_column.indicator_column(age),
##      tf.feature_column.embedding_column(consumption_ability, dimension=8),
  ]

  return wide_columns, deep_columns


def input_fn(data_file, num_epochs, shuffle, batch_size):
  """Generate an input function for the Estimator."""
  assert tf.gfile.Exists(data_file), (
      '%s not found. Please make sure you have run census_dataset.py and '
      'set the --data_dir argument to the correct path.' % data_file)

  def parse_csv(value):
    tf.logging.info('Parsing {}'.format(data_file))
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    features = dict(zip(_CSV_COLUMNS, columns))
    # features = {"bid": csv_decode_obj, "show": csv_decode_obj, ...}
    features["age"] = tf.string_split(columns[4:5], delimiter=":")
    features["education"] = tf.string_split(columns[5:6], delimiter=":")
    features["consumption_ability"] = tf.string_split(columns[6:7], delimiter=":")
    labels = features.pop('show')
    return features, labels
#    classes = tf.equal(labels, 0.3)  # binary classification
#    return features, classes

  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TextLineDataset(data_file)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

  dataset = dataset.map(parse_csv, num_parallel_calls=5)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  return dataset


