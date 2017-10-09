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
from dataset import Dataset

# Arguments for Dataset class
csv_fpath = "../configs/CMU_MOSI_all.csv"
timestamps = "relative" # absolute or relative, relative will output features relative to segment time 

# Code for loading
d = Dataset(csv_fpath, timestamps=timestamps)
features = d.load()

# View modalities
print(d.modalities) # Modalities are numbered as modality_0, modality_1, ....

# load the labels
print("Loading labels...")
labels_dict = defaultdict(lambda: dict())
label_table = pd.read_csv("../datasets/MOSI/labels/OpinionLevelSentiment.csv", header=None)
for i in range(label_table.shape[0]):
    vid = label_table.iloc[i][2]
    sid = str(label_table.iloc[i][3]) # in the feature dict sid is a number in string format
    label = (label_table.iloc[i][4] > 0) # for this tutorial we only predict positive or negative
    labels_dict[vid][sid] = label
print("Finished!")


# Some data preprocessing
maxlen = 15 # Each utterance should not have more than 15 words
video_count = 0
x_train = []
y_train = []
x_test = []
y_test = []

print("Preparing train and test data...")
# By looking at the d.modalities we can know that modality_3 is the embeddings
for vid, vdata in features['modality_3'].items():
    video_count += 1 # keep track of how many videos we have seen, only the first 63 used for train
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
        label = labels_dict[vid][sid]

        if video_count <= 63:
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
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
batch_size = 32

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=20,
          validation_data=[x_test, y_test])
