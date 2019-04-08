# -*- coding: utf-8 -*-

from random import randint
import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical

class Generator:
  def __init__(self, label_size, file=None):
    self.label_size = label_size
    self.labels = pd.read_csv('./csvfiles/labels.csv', header=None)[0].values.tolist()
    if file:
      data = np.load(file)
      self.x, self.y = data['x'], data['y']
      self.size = data['x'].shape[0]

  def generate_batch(self):
    idx = randint(0, self.size-1)
    x, y = self.x[idx], self.y[idx]
    label = self.labels.index(y)
    return x, to_categorical(label, num_classes=self.label_size)

  def generate(self, batch_size):
    while True:
      batch_x = []
      batch_y = []
      for i in range(0, batch_size):
        x, y = self.generate_batch()
        batch_x.append(x)
        batch_y.append(y)
      yield np.array(batch_x), np.array(batch_y)
