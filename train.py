# coding:utf-8
# Produced by Jiaxin Zhang
# Start Data: 20_May_2020
# Train the model.
#
# For GOD I Trust.
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

import argparse
import functools
import tf_metrics
import tensorflow as tf
from pathlib import Path

from EANN import EANNModel
from data_utils import train_input_fn
from data_utils import eval_input_fn
from data_utils import server_input_fn
import config as _cg
from setup import Setup
from log import log_info as _info
from log import log_error as _error
setup = Setup()

def learning_rate_decay(lr, cur_step, train_step):
	"""This is the original learning rate decay method from the paper."""
	p = tf.cast(cur_step, tf.float32) / tf.cast(train_step, tf.float32)
	lr = lr / (1. + 10 * p) ** 0.75
	return lr

def model_fn_builder(config):
	"""Returns 'model_fn' closure for Estimator."""
	def model_fn(features, labels, mode, params):
		# obtain the data
		_info('*** Features ***')
		for name in sorted(features.keys()):
			tf.logging.info(' name = %s, shape = %s' % (name, features[name].shape))
		
		input_text = features['input_text']
		input_image = features['input_image']

		# build the model
		is_training = (mode == tf.estimator.ModeKeys.TRAIN)
		model = EANNModel(config,
											is_training=is_training,
											input_text=input_text,
											input_image=input_image)
		label_output = model.get_label_output()
		domain_output = model.get_domain_output()
		batch_size = model.get_batch_size()

		# make predict
		label_output_prob = tf.nn.softmax(label_output, axis=-1)
		predict_label = tf.argmax(label_output_prob, axis=-1)
		domain_output_prob = tf.nn.softmax(domain_output, axis=-1)
		predict_event = tf.argmax(domain_output_prob, axis=-1)

		if mode == tf.estimator.ModeKeys.PREDICT:
			predictions = {'predict_label': predict_label,
										 'predict_event': predict_event,
										 'label_output_prob': label_output_prob,
										 'domain_output_prob': domain_output_prob}
			output_spec = tf.estimator.EstimatorSpec(mode, predictions=predictions)
		else:
			# get golden data
			classify_labels = labels['label']
			event_labels = labels['event']
			batch_size = tf.cast(batch_size, tf.float32)
			# # add l2 loss, no need
			# tv = tf.trainable_variables()
			# l2_loss = 1e-2 * (tf.reduce_sum([tf.nn.l2_loss(v) for v in tv]) / batch_size)
			
			# loss for classification
			classify_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
				labels=classify_labels,
				logits=label_output)) / batch_size
			
			# loss for event
			event_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
				labels=event_labels,
				logits=domain_output)) / batch_size
			
			# plus the 'event_loss' is correct, as add reversal layer after encoder,
			# when do backpropagation, the flag will change to minus when updating the parameters of the encoder.
			loss = classify_loss + event_loss
			
			if mode == tf.estimator.ModeKeys.TRAIN:
				# specify the learning rate
				if _cg.LEARNING_RATE_METHOD == 'polynomial':
					learning_rate = tf.train.polynomial_decay(_cg.LEARNING_RATE,
																										tf.train.get_or_create_global_step(),
																										_cg.TRAIN_STEPS,
																										end_learning_rate=_cg.LEARNING_LIMIT,
																										power=1.0,
																										cycle=False)
				elif _cg.LEARNING_RATE_METHOD == 'paper':
					learning_rate = learning_rate_decay(_cg.LEARNING_RATE, 
																							tf.train.get_or_create_global_step(),
																							_cg.TRAIN_STEPS)
				else:
					raise NotImplementedError('Not support the {}'.format(_cg.LEARNING_RATE_METHOD))
				lr = tf.maximum(tf.constant(_cg.LEARNING_LIMIT), learning_rate)

				# update the parameters
				optimizer = tf.train.GradientDescentOptimizer(lr, name='optimizer')
				tvars = tf.trainable_variables()
				gradients = tf.gradients(loss, tvars, colocate_gradients_with_ops=_cg.colocate_gradients_with_ops)
				clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
				train_op = optimizer.apply_gradients(zip(clipped_gradients, tvars), global_step=tf.train.get_global_step())
				
				# check the accuracy during training
				predict_label = tf.cast(predict_label, tf.int32)
				predict_event = tf.cast(predict_event, tf.int32)
				classify_labels = tf.cast(classify_labels, tf.int32)
				event_labels = tf.cast(event_labels, tf.int32)

				accuracy_classify = tf.reduce_mean(tf.cast(tf.equal(predict_label, classify_labels), tf.float32))
				accuracy_event = tf.reduce_mean(tf.cast(tf.equal(predict_event, event_labels), tf.float32))

				# specify the information while training
				logging_hook = tf.train.LoggingTensorHook({'step': tf.train.get_global_step(),
																									 'class_loss': classify_loss,
																									 'class_acc': accuracy_classify,
																									 'event_loss': event_loss,
																									 'event_acc': accuracy_event,
																									 'lr': learning_rate}, every_n_iter=2)
				output_spec = tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])
			elif mode == tf.estimator.ModeKeys.EVAL:
				# evaluate metrics
				metric_dict = {'accuracy': tf.metrics.accuracy(labels=classify_labels, predictions=predict_label),
											 'precision': tf_metrics.precision(labels=classify_labels, predictions=predict_label, num_classes=2),
											 'recall': tf_metrics.recall(labels=classify_labels, predictions=predict_label, num_classes=2),
											 'f1': tf_metrics.f1(classify_labels, predict_label, num_classes=2)}
				output_spec = tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metric_dict)
			else:
			  raise NotImplementedError
			
		return output_spec
	
	return model_fn

def main():
	Path(_cg.SAVE_MODEL_PATH).mkdir(exist_ok=True)

	model_fn = model_fn_builder(_cg.EANNConfig)

	gpu_config = tf.ConfigProto()
	gpu_config.gpu_options.allow_growth = True

	run_config = tf.estimator.RunConfig(		
		session_config=gpu_config,
		keep_checkpoint_max=1,
		save_checkpoints_steps=10,
		model_dir=_cg.SAVE_MODEL_PATH)

	# # For TPU		
	# run_config = tf.contrib.tpu.RunConfig(
	# 	session_config=gpu_config,
	# 	keep_checkpoint_max=1,
	# 	save_checkpoints_steps=10,
	# 	model_dir=_cg.SAVE_MODEL_PATH)

	estimator = tf.estimator.Estimator(model_fn, config=run_config)
	
	input_fn_train = functools.partial(train_input_fn, data_path=_cg.TRAIN_DATA_PATH, steps=_cg.TRAIN_STEPS)
	input_fn_eval = functools.partial(eval_input_fn, data_path=_cg.EVAL_DATA_PATH, steps=_cg.EVAL_STEPS)
	train_spec = tf.estimator.TrainSpec(input_fn=input_fn_train)
	eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_eval)
	
	tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

def package_model(ckpt_path, pb_path):
	"""Deploy the pb model, which supports plug-in inference without any other steps."""
	model_fn = model_fn_builder(_cg.EANNConfig)
	estimator = tf.estimator.Estimator(model_fn, ckpt_path)
	estimator.export_saved_model(pb_path, server_input_fn)

def parse_arguments(parser):
	parser.add_argument('-m', '--mode', type=str, help='Select mode from \'train\' or \'package\'', choices=['train', 'package'])
	return parser

if __name__ == '__main__':
	parse = argparse.ArgumentParser()
	parser = parse_arguments(parse)
	args = parser.parse_args()

	if args.mode == 'train':
		main()
	elif args.mode == 'package':
		package_model(str(_cg.SAVE_MODEL_PATH), str(_cg.PACKAGE_MODEL_PATH))