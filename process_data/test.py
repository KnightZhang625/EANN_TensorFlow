# coding:utf-8

import codecs
import pickle
import numpy as np
import process_data_weibo as process_data


def save_data(data, name):
  with codecs.open(name, 'wb') as file:
    pickle.dump(data, file)

max_ = -1
def create_data(data, name, vocab_idx, c, max_):
  text_datas = list(data['post_text'])
  image_datas = data['image']
  label_datas = list(data['label'])
  event_label_datas = list(data['event_label'])
  datas = []
  for text, image, label, event in zip(text_datas, image_datas, label_datas, event_label_datas):
    image = np.array(image)
    b, w, h = image.shape[0], image.shape[1], image.shape[2]
    image = np.reshape(image, (w, h, b))

    # test_str = ''.join(text.split(' '))
    test_str = text.split(' ')
    if len(test_str) > max_:
      max_ = len(test_str)
    for v in test_str:
      if v not in vocab_idx:
        vocab_idx[v] = c
        c +=1
    datas.append((text, image, label, event))

  print(len(datas))

  save_data(datas, name)
  return vocab_idx, c, max_

train, validate, test = process_data.get_data(False)

vocab_idx = {}
c = 0
vocab_idx, c, max_ = create_data(train, '../data_new/train_data.bin', vocab_idx, c, max_)
print(len(vocab_idx), c)
vocab_idx, c, max_ = create_data(validate, '../data_new/validate_data.bin', vocab_idx, c, max_)
print(len(vocab_idx), c)
vocab_idx, c, max_ = create_data(test, '../data_new/test_data.bin', vocab_idx, c, max_)
print(len(vocab_idx), c)

print(c)
vocab_idx['<unk>'] = c
vocab_idx['<padding>'] = c + 1
with codecs.open('../data_new/vocab_idx_new.pt', 'wb') as file:
  pickle.dump(vocab_idx, file)
print(len(vocab_idx))
print(max_)
# print(type(train))
# print(len(train))
# dict_keys(['post_text', 'original_post', 'image', 'social_feature', 'label', 'event_label', 'post_id', 'image_id']
# post_text: [43, ]
# image: list with length 43, [3, 224, 224]
# label: [43, ]
# print(len(test['image']))
# print(train['post_text'][0])
# print(type(train['social_feature'][0]))
# print(train['social_feature'][0].size())

