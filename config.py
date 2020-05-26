# coding:utf-8
# Produced by Jiaxin Zhang
# Start Data: 20_May_2020
# Configuration for the model.
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

import sys
from pathlib import Path
PROJECT_PATH = Path(__file__).absolute().parent
sys.path.insert(0, str(PROJECT_PATH))

from log import log_info as _info
from log import log_error as _error

TRAIN_DATA_PATH = PROJECT_PATH / 'data_new/train_data.bin'
EVAL_DATA_PATH = PROJECT_PATH / 'data_new/validate_data.bin'
TEST_DATA_PATH = PROJECT_PATH / 'data_new/test_data.bin'
VOCAB_PATH = PROJECT_PATH / 'data_new/vocab_idx_new.pt'

SAVE_MODEL_PATH = PROJECT_PATH / 'models'
PACKAGE_MODEL_PATH = PROJECT_PATH / 'pb_models'

BATCH_SIZE = 20
LEARNING_RATE = 5e-2
LEARNING_LIMIT = 1e-3
LEARNING_RATE_METHOD = 'paper'   # 'polynomial' or 'paper', where paper refers to the original method from the paper
colocate_gradients_with_ops = True
TRAIN_STEPS = 10000
EVAL_STEPS = 1

VGG_19_Weights_Download_URL = '''https://github-production-release-asset-2e65be.s3.amazonaws.com
/64878964/b0a81400-5983-11e6-8d11-beae6f3297b5?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=
AKIAIWNJYAX4CSVEH53A%2F20200523%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200523T051704Z&X-Amz-
Expires=300&X-Amz-Signature=7706b701ef49bdf90edde13d857945e3bb23c755bd538ac8a34206640fee1e49&X-Amz-
SignedHeaders=host&actor_id=36263466&repo_id=64878964&response-content-disposition=attachment%3B%20
filename%3Dvgg19_weights_tf_dim_ordering_tf_kernels_notop.h5&response-content-type=application%2Foctet-stream
'''

def forbid_new_attributes(wrapped_setatrr):
  def __setattr__(self, name, value):
      if hasattr(self, name):
          wrapped_setatrr(self, name, value)
      else:
          _error('Add new {} is forbidden'.format(name))
          raise AttributeError
  return __setattr__

class NoNewAttrs(object):
  """forbid to add new attributes"""
  __setattr__ = forbid_new_attributes(object.__setattr__)
  class __metaclass__(type):
      __setattr__ = forbid_new_attributes(type.__setattr__)

class EANNConfig(NoNewAttrs):
  # Text
  vocab_size = 1952
  embedding_size = 320 
  window_size = [1, 2, 3, 4]
  filter_number_text = 20
  max_length = 78
  pool_size = [78, 77, 76, 75]
  
  # Global
  hidden_size = 128
  num_classes = 2
  num_domains = 10

  initializer_range = 0.01
  dropout = 0.0
  
eann_config = EANNConfig()

print(VGG_19_Weights_Download_URL)