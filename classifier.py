# -*- coding: utf-8 -*-

import argparse
import numpy as np
from scipy.spatial import distance_matrix

from lib.converter import convert, proceed
from lib.network import Network
from lib.generator import Generator
from keras.callbacks import ModelCheckpoint, TensorBoard

parser = argparse.ArgumentParser()
parser.add_argument(
  'mode',
  type=str,
  help='',
  choices=['train', 'convert' , 'summary', 'predict']
)

args = parser.parse_args()

BATCH_SIZE = 2048
HIDDEN_SIZE = 512
FEATURE_SIZE = 840
LABEL_SIZE = 137

if args.mode == 'convert':
  convert()
elif args.mode == 'summary':
  network = Network(BATCH_SIZE, HIDDEN_SIZE, FEATURE_SIZE, LABEL_SIZE)
  network.model.summary()
elif args.mode == 'train':
  network = Network(BATCH_SIZE, HIDDEN_SIZE, FEATURE_SIZE, LABEL_SIZE)
  network.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])
  generator = Generator(LABEL_SIZE, './data/train_data.npz')
  val = Generator(LABEL_SIZE, './data/val_data.npz')
  network.model.fit_generator(
    generator=generator.generate(BATCH_SIZE),
    validation_data=val.generate(BATCH_SIZE),
    validation_steps=1,
    steps_per_epoch=128,
    epochs=1024,
    callbacks=[
      ModelCheckpoint(filepath="./weights.hdf5", save_weights_only=True, save_best_only=True),
      TensorBoard(log_dir='./logs'),
    ]
  )
elif args.mode == 'predict':
  network = Network(BATCH_SIZE, HIDDEN_SIZE, FEATURE_SIZE, LABEL_SIZE)
  network.model.load_weights('./weights.hdf5')
  mat = np.nan_to_num(proceed())
  res = network.model.predict(mat, batch_size=20)
  distance = distance_matrix(res, res)
  np.savetxt("result.csv", distance, delimiter=",", fmt='%10.12f')
