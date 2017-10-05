#!/usr/bin/env python
"""
The file contains the example code for loading the dataset given the csv file
"""
from dataset_v2 import Dataset_v2

__author__ = "Prateek Vij"
__copyright__ = "Copyright 2017, Carnegie Mellon University"
__credits__ = ["Amir Zadeh", "Prateek Vij", "Soujanya Poria"]
__license__ = "GPL"


# Arguments for Dataset class
csv_fpath = "../MOSI_NOFACET_all.csv"
timestamps = "absolute" # absolute or relative, relative will output features relative to segment time

# Code for loading
d = Dataset_v2(csv_fpath, timestamps=timestamps)
features = d.load()

# View modalities
print d.modalities # Modalities are numbered as modality_0, modality_1, ....

# View features for a particular segment of a modality
modality = "modality_1" # replace 0 with 1, 2, .... for different modalities
video_id = '_dI--eQ6qVU' # example video_id
segment_id = '2' # sample segment_id

#for feat in features[modality][video_id][segment_id]:
#	print feat[1] # tuples of form (start_time, end_time, feat_val)

#print len(features[modality][video_id][segment_id][3][2])

print features["modality_2"]["d6hH302o4v8"]['25']

#d.align(modality)
#print d.modalities

