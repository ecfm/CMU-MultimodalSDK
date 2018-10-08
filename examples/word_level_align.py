#word_level_align.py
#first aligns a dataset to the words vectors and collapses other modalities (by taking average of them for the duration of the word). After this operation every modality will have the same frequency (same as word vectors). Then the code aligns based on opinion labels (note that collapse does not happen for this step.

import mmsdk
from mmsdk import mmdatasdk
import numpy


def myavg(intervals,features):
        return numpy.average(features,axis=0)

cmumosi_highlevel=mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.highlevel,'cmumosi/')
cmumosi_highlevel.align('glove_vectors',collapse_functions=[myavg])
cmumosi_highlevel.add_computational_sequences(mmdatasdk.cmu_mosi.labels,'cmumosi/')
cmumosi_highlevel.align('Opinion Segment Labels')

deploy_files={x:x for x in cmumosi_highlevel.computational_sequences.keys()}

cmumosi_highlevel.deploy("./deployed",deploy_files)

aligned_cmumosi_highlevel=mmdatasdk.mmdataset('./deployed')


