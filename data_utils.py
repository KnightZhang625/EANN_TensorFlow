# coding:utf-8
# Produced by Jiaxin Zhang
# Start Data: 20_May_2020
# Data processing module.
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
import random
import codecs
import pickle
import functools
import tensorflow as tf
tf.enable_eager_execution()
import config

def load_dict():
  global vocab_idx
  with codecs.open(config.VOCAB_PATH, 'rb') as file:
    vocab_idx = pickle.load(file)
load_dict()

def create_batch_idx(data_length, batch_size):
  batch_number = data_length // batch_size
  batch_number = batch_number if data_length % batch_size == 0 else batch_number + 1

  for i in range(batch_number):
    yield (i * batch_size, i * batch_size + batch_size)

# function for converting string to idx
convert_vocab_idx = lambda string : [vocab_idx[v] if v in vocab_idx else vocab_idx['<unk>'] \
                                      for v in string.split(' ')]
# padding
padding_func = lambda data, max_length : data + [vocab_idx['<padding>'] \
                                                  for _ in range(max_length - len(data))]
padding_func_with_args = functools.partial(padding_func, max_length=config.eann_config.max_length)

def data_generator(data_path):
  # load the data
  with codecs.open(data_path, 'rb') as file:
    datas = pickle.load(file)

  # shuffle the data
  datas = copy.deepcopy(datas)  
  random.shuffle(datas)

  # generate batch data
  batch_size = config.BATCH_SIZE
  for (start, end) in create_batch_idx(len(datas), batch_size):
    data_batch = datas[start : end]
    text = [data[0] for data in data_batch]
    image = [data[1] for data in data_batch]
    label = [data[2] for data in data_batch]
    event = [data[3] for data in data_batch]

    # convert string to idx and paddinng
    text_idx = list(map(convert_vocab_idx, text))
    text_idx_padded = list(map(padding_func_with_args, text_idx))

    features = {'input_text': text_idx_padded,
                'input_image': image}
    tags = {'label': label,
            'event': event}
    yield(features, tags)

def input_fn(func):
  @functools.wraps(func)
  def input_fn(data_path, steps):
    output_types = {'input_text': tf.int32,
                    'input_image': tf.float32}
    output_shapes = {'input_text': [None, None],
                     'input_image': [None, None, None, 3]}
    tag_types = {'label': tf.int32,
                 'event': tf.int32}
    tag_shapes = {'label': [None],
                  'event': [None]}
    
    data_generator_with_path = functools.partial(data_generator, data_path=data_path)
    dataset = tf.data.Dataset.from_generator(
      data_generator_with_path,
      output_types=(output_types, tag_types),
      output_shapes=(output_shapes, tag_shapes))
    
    dataset = dataset.repeat(steps)
    return dataset
  
  return input_fn

@input_fn
def train_input_fn():
  pass

@input_fn
def eval_input_fn():
  pass

def server_input_fn():
  input_text = tf.placeholder(tf.int32, shape=[None, None], name='input_text')
  input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')

  receive_tensor = {'input_text': input_text,
                    'input_image': input_image}
  features = {'input_text': input_text,
              'input_image': input_image}
  
  return tf.estimator.export.ServingInputReceiver(features, receive_tensor)

if __name__ == '__main__':
  for datas in train_input_fn(config.TRAIN_DATA_PATH, 10):
    print(datas[1]['label'])
    print(datas[1]['event'])
    input()