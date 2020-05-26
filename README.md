# EANN_TensorFlow

##### *Produced by Jiaxin Zhang*

*This is the TensorFlow implementation for the paper &lt;&lt;EANN: Event Adversarial Neural Networks for Multi-Modal Fake News Detection>>, including industrial deployment.*

## Requirements
- python 3
- tensorflow == 1.14 (*prefer to 1.14, >1.12 also satisfied*)
-  *If encounter mismatch GPU Driver problem, please install tensorflow_gpu by ```conda install tensorflow-gpu=1.14```*

## Run Example
```shell
# train
python train.py -m train
# package model for inference
python train.py -m package
# predict
python predict.py
```

## Description
-  **data_new**
>> The train, evaluate, test datas are saved here.
-  **process_data**
>> Make data, no need to modify if only run the example.
-  **<a  href="https://github.com/KnightZhang625/EANN_TensorFlow/blob/master/EANN.py">EANN.py</a>**
>> The main EANN model.
-  **<a  href="https://github.com/KnightZhang625/EANN_TensorFlow/blob/master/config.py">config.py</a>**
>> Revise the configuration in this file.
-  **<a  href="https://github.com/KnightZhang625/EANN_TensorFlow/blob/master/data_utils.py">data_utils.py</a>**
>> The function provides data input for the model.
-  **<a  href="https://github.com/KnightZhang625/EANN_TensorFlow/blob/master/model_helper.py">model_helper.py</a>**
>> Supplenmentary functions for the main model.
-  **<a  href="https://github.com/KnightZhang625/EANN_TensorFlow/blob/master/train.py">train.py</a>**
>> Function for training and packaging the model.
-  **<a  href="https://github.com/KnightZhang625/EANN_TensorFlow/blob/master/predict.py">predict.py</a>**
>> Function for prediction the model.

## Reference
-  **<a  href="https://github.com/KnightZhang625/EANN_TensorFlow/blob/master/reversal_gradient.py">reversal_gradient.py</a>** is from https://github.com/KnightZhang625/tf-dann/blob/master/flip_gradient.py
- textCNN revised from https://github.com/dennybritz/cnn-text-classification-tf/blob/master/text_cnn.py