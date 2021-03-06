# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Train DNN on tt income dataset."""

import os

from absl import app as absl_app
from absl import flags
import tensorflow as tf

from official.utils.flags import core as flags_core
from official.utils.logs import logger
import tt_dataset
import wide_deep_run_loop


def define_tt_flags():
  wide_deep_run_loop.define_wide_deep_flags()
  flags.adopt_module_key_flags(wide_deep_run_loop)
  flags_core.set_defaults(data_dir='/tmp/tt_data',
                          model_dir='/tmp/tt_model',
                          train_epochs=20,
                          epochs_between_evals=3,
                          inter_op_parallelism_threads=0,
                          intra_op_parallelism_threads=0,
                          batch_size=4)


def build_estimator(model_dir, model_type, model_column_fn, inter_op, intra_op):
  """Build an estimator appropriate for the given model type."""
  wide_columns, deep_columns = model_column_fn()
  hidden_units = [128, 64, 32, 16]

  # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
  # trains faster than GPU for this model.
  run_config = tf.estimator.RunConfig().replace(
      session_config=tf.ConfigProto(device_count={'GPU': 0},
                                    inter_op_parallelism_threads=inter_op,
                                    intra_op_parallelism_threads=intra_op))


  if model_type == 'wide':
    return tf.estimator.LinearClassifier(
        model_dir=model_dir,
        feature_columns=wide_columns,
        config=run_config)
  elif model_type == 'deep':
    return tf.estimator.DNNClassifier(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=hidden_units,
        config=run_config)
  else:
    return tf.estimator.DNNLinearCombinedRegressor(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config)

#    return tf.estimator.DNNLinearCombinedClassifier(
#        model_dir=model_dir,
#        linear_feature_columns=wide_columns,
#        dnn_feature_columns=deep_columns,
#        dnn_hidden_units=hidden_units,
#        config=run_config)


def run_tt(flags_obj):
  """Construct all necessary functions and call run_loop.

  Args:
    flags_obj: Object containing user specified flags.
  """
  train_file = os.path.join(flags_obj.data_dir, tt_dataset.TRAINING_FILE)
  test_file = os.path.join(flags_obj.data_dir, tt_dataset.EVAL_FILE)
  submission_file = os.path.join(flags_obj.data_dir, tt_dataset.SUBMISSION_FILE)

  # Train and evaluate the model every `flags.epochs_between_evals` epochs.
  def train_input_fn():
    return tt_dataset.input_fn(
        train_file, flags_obj.epochs_between_evals, True, flags_obj.batch_size, "train")

  def eval_input_fn():
    return tt_dataset.input_fn(test_file, 1, False, flags_obj.batch_size, "eval")

  def pred_input_fn():
    return tt_dataset.input_fn(submission_file, 1, False, flags_obj.batch_size, "pred")


  tensors_to_log = {
      'average_loss': '{loss_prefix}head/truediv',
      'loss': '{loss_prefix}head/weighted_loss/Sum'
  }

  wide_deep_run_loop.run_loop(
      name="tt data", train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      pred_input_fn=pred_input_fn,
      model_column_fn=tt_dataset.build_model_columns,
      build_estimator_fn=build_estimator,
      flags_obj=flags_obj,
      tensors_to_log=tensors_to_log,
      early_stop=True)


def main(_):
  with logger.benchmark_context(flags.FLAGS):
    run_tt(flags.FLAGS)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  define_tt_flags()
  absl_app.run(main)
