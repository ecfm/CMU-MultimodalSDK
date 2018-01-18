#!/usr/bin/env python

'''
This script shows you how to:
1. Load the features for each segment;
2. Load the labels for each segment;
3. Prerocess data and use Keras to implement a simple LSTM on top of the data
'''

from __future__ import print_function
import numpy as np
import pandas as pd
from collections import defaultdict
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from mmdata import MOSI


# Download the data if not present
mosi = MOSI()
embeddings = mosi.embeddings()
sentiments = mosi.sentiments()
train_ids = mosi.train()
valid_ids = mosi.valid()
test_ids = mosi.test()

# Some data preprocessing
maxlen = 15 # Each utterance will be truncated/padded to 15 words
x_train = []
y_train = []
x_test = []
y_test = []

print("Preparing train and test data...")
for vid, vdata in embeddings['embeddings'].items(): # note that even Dataset with one feature will require explicit indexing of features
    for sid, sdata in vdata.items():
        if sdata == []:
            continue
        example = []
        for i, time_step in enumerate(sdata):
            # data is truncated for 15 words
            if i == 15:
                break
            example.append(time_step[2]) # here first 2 dims (timestamps) will not be used

        for i in range(maxlen - len(sdata)):
            example.append(np.zeros(sdata[0][2].shape)) # padding each example to maxlen
        example = np.asarray(example)
        label = 1 if sentiments[vid][sid] >= 0 else 0 # binarize the labels

        # here we just use everything except training set as the test set
        if vid in train_ids:
            x_train.append(example)
            y_train.append(label)
        else:
            x_test.append(example)
            y_test.append(label)

# Prepare the final inputs as numpy arrays
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
print("Data preprocessing finished! Begin compiling and training model.")

model = Sequential()
model.add(LSTM(64, input_shape=(maxlen, 300)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


# try using different optimizers and different optimizer configs
model.compile('adam', 'binary_crossentropy', metrics=['binary_accuracy'])
batch_size = 32

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=20,
          validation_data=[x_test, y_test])
