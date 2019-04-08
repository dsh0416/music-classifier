# -*- coding: utf-8 -*-

import pickle
import numpy as np
from lib.feature_loader import FeatureLoader
from lib.MFCC import MFCCs

# Convert pickle files to npz for backwards-compatibility
def convert():
  for file in ['train_data', 'val_data', 'test_data']:
    with open('./data/' + file + '.pickle', 'rb') as f:
      x, y, z = pickle.load(f)
      np.savez_compressed('./data/' + file + '.npz', x=x, y=y)

def proceed():
  feature_loader = FeatureLoader()
  mfcc = MFCCs()
  for i in range(1, 21):
    mfcc.get_mfccs(filename='./audio/' + str(i) + '.wav')
    rhy = feature_loader.get_op_from_melspec(mfcc.melspec, K=2).T[0]
    mel = feature_loader.get_pb_for_file('./csvfiles/melody/' + str(i) + '.csv').T[0]
    mfc = feature_loader.get_mfcc_from_melspec(mfcc.melspec).T[0]
    chroma = feature_loader.get_chroma_for_file('./csvfiles/harmony/' + str(i) + '.csv').T[0]
    print(np.concatenate((rhy, mel, mfc, chroma)).shape)
