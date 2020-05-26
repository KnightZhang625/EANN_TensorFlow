# coding:utf-8
# Produced by Jiaxin Zhang
# Start Data: 20_May_2020
# Inference for the model.
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

import codecs
import pickle
from pathlib import Path
from tensorflow.contrib import predictor

import config
from data_utils import create_batch_idx, convert_vocab_idx, padding_func_with_args

def restore_model(pb_path):
  """Restore the latest model from the given path."""
  subdirs = [x for x in Path(pb_path).iterdir()
             if x.is_dir() and 'temp' not in str(x)]
  latest_model = str(sorted(subdirs)[-1])
  predict_fn = predictor.from_saved_model(latest_model)

  return predict_fn

def predict(model, test_path, batch_size):
  with codecs.open(test_path, 'rb') as file:
    test_data = pickle.load(file)
  
  classify_results = []
  event_results = []
  for (start, end) in create_batch_idx(len(test_data), batch_size):
    data_batch = test_data[start : end]
    text = [data[0] for data in data_batch]
    image = [data[1] for data in data_batch]
    label = [data[2] for data in data_batch]
    event = [data[3] for data in data_batch]

    print(label, event)

    # convert string to idx and paddinng
    text_idx = list(map(convert_vocab_idx, text))
    text_idx_padded = list(map(padding_func_with_args, text_idx))

    features = {'input_text': text_idx_padded,
                'input_image': image}
    
    predictions = model(features)
    classify_labels = predictions['predict_label']
    event_labels = predictions['predict_event']
    classify_probs = predictions['label_output_prob']
    event_probs = predictions['domain_output_prob']
    print(classify_labels)
    print(event_labels)
    print(classify_probs)
    print(event_probs)
    classify_results.extend(classify_labels)
    event_results.extend(event_labels)
  
  return classify_results, event_results

if __name__ == '__main__':
  model = restore_model(config.PACKAGE_MODEL_PATH)

  classify_results, event_results = predict(model, config.TEST_DATA_PATH, 3)
  print(classify_results)
  print(event_results)