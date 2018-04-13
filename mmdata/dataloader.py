from __future__ import print_function, division
import os
import sys
from .dataset import Dataset
from pickle import load
from .utils import download

def compatible_load(file_handle):
    '''
    A loading function that is compatible with python 2 and 3
    '''
    if sys.version_info >= (3, 5):
        return load(file_handle, encoding='bytes')
    else:
        return load(file_handle)

def convert_features(feature):
    '''
    Convert a Dataset object from python 2 pickle to python 3 pickle
    '''
    # convert dunder keys
    dunder_keys = list(feature.__dict__.keys())
    for key in dunder_keys:
        feature.__dict__[key.decode('utf-8')] = feature.__dict__.pop(key)

    # convert phonemes_dict
    feature.phoneme_dict = list(map(lambda x: x.decode('utf-8'), feature.phoneme_dict))

    # convert the modalities info dict
    for modality in list(feature.modalities.keys()):
        for column in list(feature.modalities[modality].keys()):
            feature.modalities[modality][column.decode('utf-8')] = feature.modalities[modality].pop(column).decode('utf-8')
        feature.modalities[modality.decode('utf-8')] = feature.modalities.pop(modality)

    # convert the feature dict
    for modality in list(feature.feature_dict.keys()):
        for vid in list(feature.feature_dict[modality].keys()):
            for sid in list(feature.feature_dict[modality][vid].keys()):
                feature.feature_dict[modality][vid][sid.decode('utf-8')] = feature.feature_dict[modality][vid].pop(sid)
            feature.feature_dict[modality][vid.decode('utf-8')] = feature.feature_dict[modality].pop(vid)
        feature.feature_dict[modality.decode('utf-8')] = feature.feature_dict.pop(modality)
    return feature

def convert_labels(labels):
    '''convert the labels dict'''
    for vid in list(labels.keys()):
        for sid in list(labels[vid].keys()):
            labels[vid][sid.decode('utf-8')] = labels[vid].pop(sid)
        labels[vid.decode('utf-8')] = labels.pop(vid)
    return labels

def convert_partition(id_file):
    '''convert the partitions set'''
    _id_file = set()
    for vid in id_file:
        _id_file.add(vid.decode('utf-8'))
    id_file = _id_file
    return id_file


class Dataloader(object):
    """Loader object for datasets"""
    def __init__(self, dataset_url):
        self.path = os.path.abspath(__file__)
        self.folder = self.path.replace("dataloader.pyc", "").replace("dataloader.py", "")
        self.dataset_folder = dataset_url.split('/')[-1]
        self.location = os.path.join(self.folder, 'data', self.dataset_folder, 'pickled')
        self.dataset = dataset_url.lstrip('/')
    
    def get_feature(self, feature):
        """The unified API for getting specified features"""
        feature_path = os.path.join(self.location, feature + '.pkl')
        feature_present = os.path.exists(feature_path)
        if not feature_present:
            downloaded = download(self.dataset, feature, self.location)
            if not downloaded:
                return None

        # TODO: check MD5 values and etc. to ensure the downloaded dataset's intact
        with open(feature_path, 'rb') as fp:
            try:
                feature_values = compatible_load(fp)
            except:
                print("The previously downloaded dataset is compromised, downloading a new copy...")
                downloaded = download(self.dataset, feature, self.location)
                if not downloaded:
                    return None
                else:
                    feature_values = compatible_load(fp)
        return feature_values

    def facet(self):
        """Returns a single-field dataset object for facet features"""
        facet_values = self.get_feature('facet')
        if sys.version_info >= (3, 5):
            facet_values = convert_features(facet_values)
        return facet_values

    def openface(self):
        """Returns a single-field dataset object for openface features"""
        openface_values = self.get_feature('openface')
        if sys.version_info >= (3, 5):
            openface_values = convert_features(openface_values)
        return openface_values

    def embeddings(self):
        """Returns a single-field dataset object for embeddings"""
        embeddings_values = self.get_feature('embeddings')
        if sys.version_info >= (3, 5):
            embeddings_values = convert_features(embeddings_values)
        return embeddings_values

    def words(self):
        """Returns a single-field dataset object for one-hot vectors of words"""
        words_values = self.get_feature('words')
        try:
            words_values = convert_features(words_values) # ----> words.pkl are processed in a way we don't have to convert
        except:
            pass # later need to specify the Error type here
        return words_values

    def phonemes(self):
        """Returns a single-field dataset object for one-hot vectors of phonemes"""
        phonemes_values = self.get_feature('phonemes')
        if sys.version_info >= (3, 5):
            phonemes_values = convert_features(phonemes_values)
        return phonemes_values

    def covarep(self):
        """Returns a single-field dataset object for covarep features"""
        covarep_values = self.get_feature('covarep')
        if sys.version_info >= (3, 5):
            covarep_values = convert_features(covarep_values)
        return covarep_values

    def opensmile(self):
        """Returns a single-field dataset object for opensmile features"""
        opensmile_values = self.get_feature('opensmile')
        if sys.version_info >= (3, 5):
            opensmile_values = convert_features(opensmile_values)
        return opensmile_values

    def sentiments(self):
        """Returns a nested dictionary that stores the sentiment values"""
        sentiments_values = self.get_feature('sentiments')
        if sys.version_info >= (3, 5):
            sentiments_values = convert_labels(sentiments_values)
        return sentiments_values

    def emotions(self):
        """Returns a nested dictionary that stores the emotion distributions"""
        emotions_values = self.get_feature('emotions')
        if sys.version_info >= (3, 5):
            emotions_values = convert_labels(emotions_values)
        return emotions_values

    def train(self):
        """Returns three sets of video ids: train, dev, test"""
        train_ids = self.get_feature('train')
        if sys.version_info >= (3, 5):
            train_ids = convert_partition(train_ids)
        return train_ids

    def valid(self):
        """Returns three sets of video ids: train, dev, test"""
        valid_ids = self.get_feature('valid')
        if sys.version_info >= (3, 5):
            valid_ids = convert_partition(valid_ids)
        return valid_ids

    def test(self):
        """Returns three sets of video ids: train, dev, test"""
        test_ids = self.get_feature('test')
        if sys.version_info >= (3, 5):
            test_ids = convert_partition(test_ids)
        return test_ids

    def original(self, dest):
        """Downloads the dataset files as a tar ball, to the specified destination"""
        # raw_path = os.path.join(dest, self.dataset + '.tar')
        # downloaded = download_raw(self.dataset, dest)
        raise NotImplementedError


class MOSI(Dataloader):
    """Dataloader for CMU-MOSI dataset"""
    def __init__(self):
        super(MOSI, self).__init__('http://sorena.multicomp.cs.cmu.edu/downloads/MOSI')
        print("This API will be deprecated in the future versions. Please check the Github page for the current API")


class MOSEI(Dataloader):
    """Dataloader for CMU-MOSEI dataset"""
    def __init__(self):
        super(MOSEI, self).__init__('http://sorena.multicomp.cs.cmu.edu/downloads/MOSEI')
        print("This API will be deprecated in the future versions. Please check the Github page for the current API")
