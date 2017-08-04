#!/usr/bin/env python
"""
The file contains the class and methods for loading textual features
from P2FA files. Phonemes and words are loaded as one-hot embeddings
"""
import numpy as np 
import pandas as pd
from os import system
from os.path import join

__author__ = "Prateek Vij"
__copyright__ = "Copyright 2017, Carnegie Mellon University"
__credits__ = ["Amir Zadeh", "Prateek Vij", "Soujanya Poria"]
__license__ = "GPL"
__version__ = "1.0.1"
__status__ = "Production"

class P2FA_Helper():
    """
    Class for loading words, embeddings and phonemes as features from 
    P2FA files
    """

    def __init__(self, p2fa_csv, output_dir="./", embed_type = "w2v",
                embed_dict_path=None):
        """
        Initialise P2FA helper class.
        :param p2fa_csv: Path to csv file containing fpaths of p2fa files
        :param output_dir: Path to the output directory to store computed
                features. Can be either a string or a list of 3 strings 
                ( 2 if embed_dict_path is None). If string, subdirectories
                shall be created to store different features
        :param embed_type: Embeddding type - glove or w2v depending upon the
                type of embedding dictionary you provide in embed_dict_path
        :param embed_dict_path: Path to the embedding dictionary
        return None
        """
        self.p2fa_csv = p2fa_csv
        self.output_dir = output_dir
        self.embed_type = embed_type
        self.embed_dict_path = embed_dict_path
        self.vocabulary = []
        self.feat_count = 2
        self.dataset_info = {}
        self.feat_dict = []
        
        if self.embed_dict_path:
            self.feat_count += 1

        if embed_type not in ( "w2v", "glove"):
            raise NameError("Param embed_type must be 'w2v' or 'glove'")

        
        if isinstance(output_dir, str):
            self.embedding_dir = join(output_dir,"embeddings")
            self.words_dir = join(output_dir, "words")
            self.phonemes_dir = join(output_dir, "phonemes")
        elif (isinstance(output_dir,list) 
                and len(output_dir) == self.feat_count
                and all(isinstance(n,str) for n in output_dir) ):
            self.embedding_dir = output_dir[0]
            self.words_dir = output_dir[1]
            self.phonemes = output_dir[2]
        else:
            raise TypeError("Invalid value for the param output_dir")
        return

    def load(self):
        """
        Calls method validate_csv, compute phonemes, words, and
        word_embedding features and store them.
        :return feature dictionary for phonemes, words, and embeddings
        """
        self.validate_csv()
        phonemes_feat_dict = self.load_phonemes()
        words_feat_dict = self.load_words()
        self.feat_dict = [phonemes_feat_dict, words_feat_dict]
        if self.embed_dict_path:
            if self.embed_type == "w2v":
                embed_feat_dict = self.load_w2v()
            else:
                embed_feat_dict = self.load_glove()
            self.feat_dict.append(embed_feat_dict)
        return self.feat_dict


    def validate_csv(self):
        """
        Validate the csv file format.
        :raise Exception if file format is not correct
        :returns None
        """
        data = pd.read_csv(self.p2fa_csv, header=None)
        data = np.asarray(data)
        for record in data[2:]:
            video_id = str(record[0])
            segment_id = str(record[1])
            if video_id not in self.dataset_info:
                self.dataset_info[video_id] = {}
            if segment_id in self.dataset_info[video_id]:
                raise NameError("Multiple instances of segment "
                                +segment_id+" for video "+video_id)
            segment_data = {}
            segment_data["start"] = float(record[2])
            segment_data["end"] = float(record[3])
            segment_data["p2fa_file"] = str(record[4])
            self.dataset_info[video_id][segment_id] = segment_data
        return

    def load_phonemes(self):
        """
        Load phonemes as one-hot embeddings from P2FA files and store them
        in the directory path mentioned in self.phonemes_dir.
        :returns segment wise feature dictionary for phoneme
        """
        return None

    def load_words(self):
        """
        Load words as one-hot embeddings from P2FA files and store them
        in the directory path mentioned in self.words_dir.
        :returns segment wise feature dictionary for words
        """
        return None

    def load_w2v(self):
        """
        Load Word2Vec embeddings from P2FA files and pre-trained Word2Vec 
        KeyedVectors binary file and store them in the 
        directory path mentioned in self.embedding_dir.
        :returns segment wise feature dictionary for embeddings
        """
        return None

    def load_glove(self):
        """
        Loads glove embeddings from P2FA files and gloVe Vector file and 
        store them in directory path mentioned in self.embedding_dir.
        :returns segment wise feature dictionary for embeddings
        """
        return None

    def get_vocabulary(self):
        """
        Return Vocabulary. Must be called after calling method load or 
        load_words
        :returns: list of vocabulary words in the dataset

        """
        return None




