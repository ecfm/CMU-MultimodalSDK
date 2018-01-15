#!/usr/bin/env python
"""
The file contains the example code for loading the P2FA code for phonemes and words given the p2FA csv file
"""
from p2fa_helper import P2FA_Helper

__author__ = "Prateek Vij"
__copyright__ = "Copyright 2017, Carnegie Mellon University"
__credits__ = ["Amir Zadeh", "Prateek Vij", "Soujanya Poria"]
__license__ = "GPL"


# Arguments for Helper class
csv_fpath = "../configs/CMU_MOSI_p2fa.csv"
phoneme_dir = "/home/prateek/sandbox/Phonemes" # Modify the path accordingly
words_dir = "/home/prateek/sandbox/Words" # Modify the path accordingly

# Code for loading, word embeddings wont be loaded
p = P2FA_Helper(csv_fpath, [phoneme_dir, words_dir])

# If you want to individually load the features and store them 
p.validate_csv()
# phoneme_feats = p.load_phonemes()
word_feats = p.load_words()

# If you want to load and store both simultaneously
#f = p.load()



# View features
video_id = '_dI--eQ6qVU' # example video_id
segment_id = '2' # sample segment_id

for feat in word_feats[video_id][segment_id]:
	print feat