# -*- coding: utf-8 -*-

from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Input, Dense

class Network:
  def __init__(self, batch_size, hidden_size, feature_size, label_size):
    self.batch_size = batch_size
    self.hidden_size = hidden_size
    self.feature_size = feature_size
    self.label_size = label_size
    self.model = self.build()
  
  def build(self):
    wav = Input(shape=(self.feature_size,), name='wav')

    hidden = Dense(self.hidden_size, activation='relu')(wav)
    hidden = Dense(self.hidden_size, activation='relu')(hidden)

    output = Dense(self.label_size, activation='softmax')(hidden)
    return Model([wav], [output])
