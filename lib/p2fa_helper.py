#!/usr/bin/env python
"""
The file contains the class and methods for loading textual features
from P2FA files. Phonemes and words are loaded as one-hot embeddings
"""
import numpy as np 
import pandas as pd
from os import system
from os.path import join
import utils

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
                embed_model_path=None, embed_model_type='text'):
        """
        Initialise P2FA helper class.
        :param p2fa_csv: Path to csv file containing fpaths of p2fa files
        :param output_dir: Path to the output directory to store computed
                features. Can be either a string or a list of 3 strings 
                ( 2 if embed_model_path is None). If string, subdirectories
                shall be created to store different features
        :param embed_type: Embeddding type - glove or w2v depending upon the
                type of embedding dictionary you provide in embed_model_path
        :param embed_model_path: Path to the embedding dictionary
        :param embed_dict_type: text or binary, valid only for word2vec model
                file.
        return None
        """
        self.p2fa_csv = p2fa_csv
        self.output_dir = output_dir
        self.embed_type = embed_type
        self.embed_model_path = embed_model_path
        self.vocabulary = []
        self.feat_count = 2
        self.dataset_info = {}
        self.feat_dict = []
        self.phonemes = utils.p2fa_phonemes
        self.embed_model = None
        self.word_dict = {}
        self.embed_model_type = embed_model_type

        if self.embed_model_path:
            self.feat_count += 1

        if embed_type not in ( "w2v", "glove", "spanish"):
            raise ValueError("Param embed_type must be 'w2v' or 'glove' or 'spanish'")

        if embed_model_type not in ("text", "binary"):
            raise ValueError("Param embed_model_type must be either text \
                              or binary")

        
        if isinstance(output_dir, str):
            self.embedding_dir = join(output_dir,"embeddings")
            self.words_dir = join(output_dir, "words")
            self.phonemes_dir = join(output_dir, "phonemes")
        elif (isinstance(output_dir,list) 
                and len(output_dir) == self.feat_count
                and all(isinstance(n,str) for n in output_dir) ):
            self.phonemes_dir = output_dir[0]            
            self.words_dir = output_dir[1]
            if self.embed_model_path:
                self.embedding_dir = output_dir[2]

        else:
            raise ValueError("Invalid value for the param output_dir")
        return

    def load(self):
        """
        Calls method validate_csv, compute phonemes, words, and
        word_embedding features and store them.
        :return feature dictionary for phonemes, words, and embeddings
        """
        self.validate_csv()
        phonemes_feat_dict = self.load_phonemes()
        print "Loaded phonemes"
        # phonemes_feat_dict = None
        words_feat_dict = self.load_words()
        print "Loaded Words"
        # words_feat_dict = None
        self.feat_dict = [phonemes_feat_dict, words_feat_dict]
        if self.embed_model_path:
            if self.embed_type == "w2v":
                embed_feat_dict = self.load_w2v()
            else:
                embed_feat_dict = self.load_glove()
            self.feat_dict.append(embed_feat_dict)
        return self.feat_dict

    def load_spanish(self):
        """
        Calls method validate_csv, compute phonemes, words, and
        word_embedding features and store them.
        :return feature dictionary for phonemes, words, and embeddings
        """
        self.validate_csv()
        #phonemes_feat_dict = self.load_phonemes()
        #print "Loaded phonemes"
        phonemes_feat_dict = None
        words_feat_dict = self.load_spanish_words()
        print "Loaded spanish Words"
        # words_feat_dict = None
        self.feat_dict = [phonemes_feat_dict, words_feat_dict]
        embed_feat_dict = self.load_spanish_wv()
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
        self.p2fa_feat_level = str(data[1][-1])
        if self.p2fa_feat_level not in ('s', 'v'):
            raise ValueError("P2FA feature level must be 's' or 'v'")
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
        in the directory path mentioned in self.phonenzmes_dir.
        :returns segment wise feature dictionary for phoneme
        """
        features = {}
        system("mkdir -p "+self.phonemes_dir)
        data = self.dataset_info
        for video_id, video_data in data.iteritems():
            video_feats = {}
            for segment_id, segment_data in video_data.iteritems():
                filepath = str(segment_data["p2fa_file"])
                start = segment_data["start"]
                end = segment_data["end"]
                level = self.p2fa_feat_level
                segment_feats = self.load_phonemes_for_seg(filepath,
                                                            start, end, level)
                video_feats[segment_id] = segment_feats
                fname = video_id+"_"+segment_id+".csv"
                fpath = join(self.phonemes_dir,fname)
                with open(fpath,"wb") as fh:
                    # Writing each feature in csv file for segment
                    for f in segment_feats:
                        f_start = str(f[0])
                        f_end = str(f[1])
                        f_val = [str(val) for val in f[2].tolist()]
                        str2write = ",".join([f_start, f_end] + f_val)
                        str2write += "\n"
                        fh.write(str2write)

            features[video_id] = video_feats
        return features

    def load_words(self):
        """
        Load words as one-hot embeddings from P2FA files and store them
        in the directory path mentioned in self.words_dir.
        :returns segment wise feature dictionary for words
        """
        word_dict = {}
        system("mkdir -p "+self.words_dir)
        data = self.dataset_info
        for video_id, video_data in data.iteritems():
            video_word_dict = {}
            for segment_id, segment_data in video_data.iteritems():
                filepath = str(segment_data["p2fa_file"])
                start = segment_data["start"]
                end = segment_data["end"]
                level = self.p2fa_feat_level
                segment_feats = self.load_words_for_seg(filepath, start,
                                                        end, level)
                words = [ str(val[2]).lower() for val in segment_feats ]
                for w in words:
                    if w not in self.vocabulary:
                        self.vocabulary.append(w)
                video_word_dict[segment_id] = segment_feats
                
            word_dict[video_id] = video_word_dict
        
        self.word_dict = word_dict

        features = {}
        for video_id, video_word_data in word_dict.iteritems():
            video_feats = {}
            for segment_id, segment_word_data in video_word_data.iteritems():
                video_feats[segment_id] = []
                for word_feat in segment_word_data:
                    start = word_feat[0]
                    end = word_feat[1]
                    value = np.zeros(len(self.vocabulary))
                    value[self.vocabulary.index(word_feat[2].lower())] = 1
                    video_feats[segment_id].append((start, end, value))
                fname = video_id+"_"+segment_id+".csv"
                fpath = join(self.words_dir, fname)
                with open(fpath,"wb") as fh:
                    # Writing each feature in csv file for segment
                    for f in video_feats[segment_id]:
                        f_start = str(f[0])
                        f_end = str(f[1])
                        f_val = [str(val) for val in f[2].tolist()]
                        str2write = ",".join([f_start, f_end] + f_val)
                        str2write += "\n"
                        fh.write(str2write)
            features[video_id] = video_feats
        return features

    def load_spanish_words(self):
        """
        Load words as one-hot embeddings from P2FA files and store them
        in the directory path mentioned in self.words_dir.
        :returns segment wise feature dictionary for words
        """
        word_dict = {}
        system("mkdir -p "+self.words_dir)
        data = self.dataset_info
        for video_id, video_data in data.iteritems():
            video_word_dict = {}
            for segment_id, segment_data in video_data.iteritems():
                filepath = str(segment_data["p2fa_file"])
                start = segment_data["start"]
                end = segment_data["end"]
                level = self.p2fa_feat_level
                segment_feats = self.load_spanish_words_for_seg(filepath, start,
                                                        end, level)
                words = [ str(val[2]).lower() for val in segment_feats ]
                for w in words:
                    if w not in self.vocabulary:
                        self.vocabulary.append(w)
                video_word_dict[segment_id] = segment_feats
                
            word_dict[video_id] = video_word_dict
        
        self.word_dict = word_dict

        features = {}
        for video_id, video_word_data in word_dict.iteritems():
            video_feats = {}
            for segment_id, segment_word_data in video_word_data.iteritems():
                video_feats[segment_id] = []
                for word_feat in segment_word_data:
                    start = word_feat[0]
                    end = word_feat[1]
                    value = np.zeros(len(self.vocabulary))
                    value[self.vocabulary.index(word_feat[2].lower())] = 1
                    video_feats[segment_id].append((start, end, value))
                fname = video_id+"_"+segment_id+".csv"
                fpath = join(self.words_dir, fname)
                with open(fpath,"wb") as fh:
                    # Writing each feature in csv file for segment
                    for f in video_feats[segment_id]:
                        f_start = str(f[0])
                        f_end = str(f[1])
                        f_val = [str(val) for val in f[2].tolist()]
                        str2write = ",".join([f_start, f_end] + f_val)
                        str2write += "\n"
                        fh.write(str2write)
            features[video_id] = video_feats
        return features

    def load_w2v(self):
        """
        Load Word2Vec embeddings from P2FA files and pre-trained Word2Vec 
        KeyedVectors text file and store them in the 
        directory path mentioned in self.embedding_dir.
        :returns segment wise feature dictionary for embeddings
        :Note: Do not provide KeyedVector file in binary format
        """
        from gensim.models.keyedvectors import KeyedVectors
        from gensim.models import Word2Vec
        
        is_binary = True if self.embed_model_type == "binary" else False
        model = KeyedVectors.load_word2vec_format(self.embed_model_path, 
                                                  binary = is_binary )
        print "Word2Vec model Loaded"
        self.embed_model = model
        self.embed_length = model.vector_size
        if not self.word_dict:
            self.load_words()
        
        features = {}
        system("mkdir -p "+self.embedding_dir)
        for video_id, video_word_data in self.word_dict.iteritems():
            video_feats = {}
            for segment_id, segment_word_data in video_word_data.iteritems():
                video_feats[segment_id] = []
                for word_feat in segment_word_data:
                    start, end, word = word_feat
                    try:
                        embed = self.embed_model[word]
                    except:
                        embed = np.zeros(self.embed_length)
                    video_feats[segment_id].append((start, end, embed))

                fname = video_id+"_"+segment_id+".csv"
                fpath = join(self.embedding_dir, fname)
                with open(fpath,"wb") as fh:
                    # Writing each feature in csv file for segment
                    for f in video_feats[segment_id]:
                        f_start = str(f[0])
                        f_end = str(f[1])
                        f_val = [str(val) for val in f[2].tolist()]
                        str2write = ",".join([f_start, f_end] + f_val)
                        str2write += "\n"
                        fh.write(str2write)
            features[video_id] = video_feats
        return features

    def load_glove(self):
        """
        Loads glove embeddings from P2FA files and gloVe Vector file and 
        store them in directory path mentioned in self.embedding_dir.
        :returns segment wise feature dictionary for embeddings
        """
        self.embed_model = {}
        self.embed_length = 0
        with open(self.embed_model_path, "r") as fh:
            for line in fh:
                splits = line.rstrip().split()
                word = splits[0]
                embedding = np.asarray([float(val) for val in splits[1:]])
                self.embed_model[word] = embedding
                if not self.embed_length:
                    self.embed_length = len(embedding)

        if not self.word_dict:
            self.load_words()
        
        features = {}
        system("mkdir -p "+self.embedding_dir)
        for video_id, video_word_data in self.word_dict.iteritems():
            video_feats = {}
            for segment_id, segment_word_data in video_word_data.iteritems():
                video_feats[segment_id] = []
                for word_feat in segment_word_data:
                    start, end, word = word_feat
                    try:
                        embed = self.embed_model[word]
                    except:
                        embed = np.zeros(self.embed_length)
                    video_feats[segment_id].append((start, end, embed))

                fname = video_id+"_"+segment_id+".csv"
                fpath = join(self.embedding_dir, fname)
                with open(fpath,"wb") as fh:
                    # Writing each feature in csv file for segment
                    for f in video_feats[segment_id]:
                        f_start = str(f[0])
                        f_end = str(f[1])
                        f_val = [str(val) for val in f[2].tolist()]
                        str2write = ",".join([f_start, f_end] + f_val)
                        str2write += "\n"
                        fh.write(str2write)
            features[video_id] = video_feats
        return features

    def load_spanish_wv(self):
        self.embed_model = {}
        self.embed_length = 0
        with open(self.embed_model_path, "r") as fh:
            i = 0
            for line in fh:
                i += 1
                if i == 1:
                    continue
                line = line.split()
                word = line[0]
                embedding = [float(x) for x in line[1:]]
                self.embed_model[word] = embedding
                if not self.embed_length:
                    self.embed_length = len(embedding)

        if not self.word_dict:
            self.load_words()

        features = {}
        system("mkdir -p "+self.embedding_dir)
        for video_id, video_word_data in self.word_dict.iteritems():
            video_feats = {}
            for segment_id, segment_word_data in video_word_data.iteritems():
                video_feats[segment_id] = []
                for word_feat in segment_word_data:
                    start, end, word = word_feat
                    try:
                        embed = self.embed_model[word]
                    except:
                        embed = np.zeros(self.embed_length)
                    video_feats[segment_id].append((start, end, embed))

                fname = video_id+"_"+segment_id+".csv"
                fpath = join(self.embedding_dir, fname)
                with open(fpath,"wb") as fh:
                    # Writing each feature in csv file for segment
                    for f in video_feats[segment_id]:
                        f_start = str(f[0])
                        f_end = str(f[1])
                        f_val = [str(val) for val in f[2]]
                        str2write = ",".join([f_start, f_end] + f_val)
                        str2write += "\n"
                        fh.write(str2write)
            features[video_id] = video_feats
        return features

    def get_vocabulary(self):
        """
        Return Vocabulary. Must be called after calling method load or 
        load_words
        :returns: list of vocabulary words in the dataset
        """
        return self.vocabulary

    def load_phonemes_for_seg(self, filepath, start, end, level):
        features = []
        if level == 's':
            with open(filepath,'r') as f_handle:
                # Read the file content after the first 12 header lines
                f_content = f_handle.readlines()[12:]

            for i in range(len(f_content)):
                line = f_content[i].strip()
                if not line:
                    continue

                # When phonemes are over, stop reading the file
                if line == '"IntervalTier"':
                    break           

                if i%3 == 0:
                    feat_start = float(line)

                elif i%3 == 1:
                    feat_end = float(line)

                else:
                    if line.startswith('"') and line.endswith('"'):
                        phoneme_val = line[1:-1]
                        feat_val = utils.phoneme_hotkey_enc(phoneme_val)    
                        features.append((max(feat_start-start, 0), feat_end-start, feat_val))
                    else:
                        raise ValueError("File format error at line "+str(i))
        else:
            # Read the file content after the first 12 header lines
            with open(filepath,'r') as f_handle:
                f_content = f_handle.readlines()[12:]

            for i in range(len(f_content)):
                line = f_content[i].strip()
                if not line:
                    continue

                # When phonemes are over, stop reading the file
                if line == '"IntervalTier"':
                    break           

                if i%3 == 0:
                    feat_start = float(line)

                elif i%3 == 1:
                    feat_end = float(line)

                else:
                    if line.startswith('"') and line.endswith('"'):
                        feat_time = feat_end - feat_start

                        # Ensuring the feature lies in the segment
                        if ((feat_start <= start and feat_end > end) 
                           or (feat_start >= start and feat_end < end)
                           or (feat_start <= start 
                              and start-feat_start < feat_time/2)
                           or (feat_start >= start
                              and end - feat_start > feat_time/2)):

                            phoneme_val = line[1:-1]
                            feat_val = utils.phoneme_hotkey_enc(phoneme_val)
                            features.append((max(feat_start-start, 0), feat_end-start, feat_val))
                    else:
                        raise ValueError("File format error at line "+str(i))

        return features

    def load_words_for_seg(self, filepath, start, end, level):
        features = []

        file_offset = 0
        header_offset = 12
        with open(filepath,'r') as f_handle:
            f_content = f_handle.readlines()[header_offset:]
            for i in range(len(f_content)):
                line = f_content[i].rstrip()
                if line == '"IntervalTier"':
                    file_offset = i + 5
        
        f_content = f_content[file_offset:]
        if level == 's':
            for i in range(len(f_content)):
                line = f_content[i].strip()
                if not line:
                    continue

                if i%3 == 0:
                    feat_start = float(line)

                elif i%3 == 1:
                    feat_end = float(line)

                else:
                    if line.startswith('"') and line.endswith('"'):
                        word = line[1:-1].lower()
                        if word == "sp":
                            continue
                        feat_val = word
                        features.append((max(feat_start-start, 0), feat_end-start, feat_val))
                    else:
                        raise ValueError("File format error at line "+str(i))
        else:
            for i in range(len(f_content)):
                line = f_content[i].strip()
                if not line:
                    continue

                if i%3 == 0:
                    feat_start = float(line)

                elif i%3 == 1:
                    feat_end = float(line)

                else:
                    if line.startswith('"') and line.endswith('"'):
                        feat_time = feat_end - feat_start

                        # Ensuring the feature lies in the segment
                        if ((feat_start <= start and feat_end > end) 
                           or (feat_start >= start and feat_end < end)
                           or (feat_start <= start 
                              and start-feat_start < feat_time/2)
                           or (feat_start >= start
                              and end - feat_start > feat_time/2)):

                            word = line[1:-1].lower()
                            if word == "sp":
                                continue
                            feat_val = word
                           
                            features.append((max(feat_start-start, 0), feat_end-start, feat_val))
                    else:
                        raise ValueError("File format error at line "+str(i))

        return features


    def load_spanish_words_for_seg(self, filepath, start, end, level):
        features = []
        file_offset = 0
        header_offset = 0
        with open(filepath,'r') as f_handle:
            f_content = f_handle.readlines()[header_offset:]
            if level == 's':
                for i in range(len(f_content)):
                    line = f_content[i].strip()
                    if line.startswith('"') and line.endswith('"'):
                        word = line[1:-1].lower()
                        feat_val = word
                    elif i%3 == 1:
                        feat_start = float(line)
                    elif i%3 == 2:
                        feat_end = float(line)
                        features.append((max(feat_start-start, 0), feat_end-start, feat_val))
        return features
        


