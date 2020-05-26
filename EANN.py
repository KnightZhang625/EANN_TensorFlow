# coding:utf-8
# Produced by Jiaxin Zhang
# Start Data: 19_May_2020
# TensorFlow Version for EANN.
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

import copy
import tensorflow as tf
import model_helper as _mh
import config as _cg
from config import eann_config
from reversal_gradient import flip_gradient

from log import log_info as _info
from log import log_error as _error

class EANNModel(object):
  """The Main Model for the EANN."""
  def __init__(self,
               config,
               is_training,
               input_text,
               input_image,
               scope=None):
      """"Constructor for EANN Model.
      
      Args:
        config: Config Object, hyparameters set.
        is_training: Boolean, whether train or not.
        input_text: tf.int32 Tensor, [batch_size, seq_length].
        input_image: tf.float32 Tensor, [batch_size, h, w, c].
      """
      # config
      config = copy.deepcopy(config)

      # textCNN config
      self.vocab_size = config.vocab_size
      self.embedding_size = config.embedding_size
      self.window_size = config.window_size
      self.pool_size = config.pool_size
      self.filter_number_text = config.filter_number_text
      self.seq_length = config.max_length

      # VGG_19
      try:
        self.vgg = tf.keras.applications.VGG19(input_shape=(224, 224, 3),
                                              include_top=False,
                                              weights='imagenet')
        _info('Successfully load the pre-trained VGG-19 weights.')
      except Exception:
        _error('Please download the VGG_19 weights from : \n{}\n, then put the file \
        into ~/.keras/models'.format(_cg.VGG_19_Weights_Download_URL))
      

      self.vgg.trainable = False  # do not train the vgg pretrained parameters

      # global config
      self.hidden_size = config.hidden_size
      self.num_classes = config.num_classes
      self.num_domains = config.num_domains

      # basic config
      self.initializer_range = config.initializer_range
      self.dropout = config.dropout
      if not is_training:
        self.dropout = 0.0
      
      # Build the Graph
      self.label_output, self.domain_output, self.batch_size = self.build(input_text, input_image)

  def build(self,
            input_text,
            input_image,
            scope=None):
    """"Build the whole graph."""
    with tf.variable_scope(scope, default_name='EANN'):
      # Embedding
      with tf.variable_scope('embeddings'):
        embedding_output, self.embedding_table = _mh.embedding_lookup(
          input_ids=input_text,
          vocab_size=self.vocab_size,
          embedding_size=self.embedding_size,
          initializer_range=self.initializer_range,
          word_embedding_name='word_embeddings')
      
      # textCNN -> [batch_size, hidden_size]
      with tf.variable_scope('textCNN'):
        text_output = textCNN(embedding_output,
                              self.seq_length,
                              self.window_size,
                              self.pool_size,
                              self.filter_number_text,
                              self.hidden_size,
                              self.dropout,
                              self.initializer_range)
      # VGG_19
      with tf.variable_scope('vgg_19'):
        image_output = self.vgg(input_image)
        # image_output.pretrained()
        batch_size = _mh.get_shape_list(image_output)[0]
        # squeeze the tensor, as the following dense layer need specified last dimension,
        # must specify the exact dimension
        image_output = tf.reshape(image_output, (batch_size, 25088))
        image_output = tf.layers.dense(image_output,
                                       self.hidden_size,
                                       activation=None,
                                       name='image_output_layer',
                                       kernel_initializer=_mh.create_initializer(initializer_range=self.initializer_range))
                                       
      # concatenate the text output with the image output
      text_image_output = tf.concat((text_output, image_output), -1)

      # label classify layer
      with tf.variable_scope('classify_label'):
        label_output = self.classify_layer(text_image_output)
      # domain classify layer
      with tf.variable_scope('classify_domain'):
        # apply reversal gradient here
        reverse_text_image_output = flip_gradient(text_image_output)
        domain_output = self.classify_domain(reverse_text_image_output)

    return label_output, domain_output, batch_size

  def classify_layer(self, inputs):
    """Classify the input as class according to the number of classes."""
    output = tf.layers.dense(inputs,
                            self.num_classes,
                            activation=None,
                            name='label_layer',
                            kernel_initializer=_mh.create_initializer(initializer_range=self.initializer_range))
    return output
  
  def classify_domain(self, inputs):
    output_prev = tf.layers.dense(inputs,
                                  self.hidden_size,
                                  activation=tf.nn.relu,
                                  name='domain_layer_prev',
                                  kernel_initializer=_mh.create_initializer(initializer_range=self.initializer_range))
    output = tf.layers.dense(output_prev,
                             self.num_domains,
                             activation=None,
                             name='domain_layer_final',
                              kernel_initializer=_mh.create_initializer(initializer_range=self.initializer_range))
    return output
  
  def get_label_output(self):
    """Interface for obtaining the label output."""
    # [batch_size, num_classes]
    return self.label_output
  
  def get_domain_output(self):
    """Interface for obtaining the domain output."""
    # [batch_size, num_domains]
    return self.domain_output
  
  def get_batch_size(self):
    """Interface for obtaining the batch size."""
    return self.batch_size
              
def textCNN(embedding, 
            seq_length,
            window_size, 
            pool_size,
            filter_number, 
            hidden_size, 
            dropout_prob,
            initializer_range,
            scope=None):
  """Apply textCNN on the embeddings.
    The code here is revised from the below url:
      https://github.com/dennybritz/cnn-text-classification-tf/blob/master/text_cnn.py
    Double Salute !
  """
  embedding_shape = _mh.get_shape_list(embedding)
  seq_length = embedding_shape[1]
  embedding_size = embedding_shape[2]
  embedded_expanded = tf.expand_dims(embedding, -1)

  pooled_outputs = []
  for i, ws in enumerate(window_size):
    with tf.variable_scope(scope, default_name='conv_{}'.format(i)):
      # Conv
      filter_shape = [ws, embedding_size, 1, filter_number]
      W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
      b = tf.Variable(tf.constant(0.1, shape=[filter_number]), name="b")
      conv = tf.nn.conv2d(embedded_expanded,
                          W,
                          strides=[1, 1, 1, 1],
                          padding="VALID",
                          name="conv")
      h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
      # MaxPool
      pooled = tf.nn.max_pool(h,
                              ksize=[1, pool_size[i], 1, 1],
                              strides=[1, 1, 1, 1],
                              padding='VALID',
                              name="pool")
      pooled_outputs.append(pooled)

  # Combine all the pooled features
  num_filters_total = filter_number * len(window_size)
  h_pool = tf.concat(pooled_outputs, 3)
  h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

  # Add dropout
  with tf.name_scope("dropout"):
    h_drop = tf.nn.dropout(h_pool_flat, keep_prob=(1-dropout_prob))
  
  # Final Output
  with tf.variable_scope('textCNN_output'):
    output = tf.layers.dense(
      h_drop,
      hidden_size,
      activation=tf.nn.relu,
      name='layer_output',
      kernel_initializer=_mh.create_initializer(initializer_range=initializer_range))
  
  return output