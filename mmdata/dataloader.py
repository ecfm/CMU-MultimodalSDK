import os
from dataset import Dataset
from cPickle import load
from utils import download

class Dataloader(object):
    """Loader object for datasets"""

    def __init__(self, dataset):
        self.path = os.path.abspath(__file__)
        self.folder = self.path.replace("dataloader.pyc", "").replace("dataloader.py", "")
        self.location = os.path.join(self.folder, 'data', dataset, 'pickled')
        self.dataset = dataset
    
    def get_feature(self, feature):
        """The unified API for getting specified features"""
        feature_path = os.path.join(self.location, feature + '.pkl')
        feature_present = os.path.exists(feature_path)
        if not feature_present:
            download(self.dataset, feature, self.location)

        # TODO: check MD5 values and etc. to ensure the downloaded dataset's intact
        with open(feature_path, 'rb') as fp:
            feature_values = load(fp)
        return feature_values

    def facet(self):
        """Returns a single-field dataset object for facet features"""
        facet_values = self.get_feature('facet')
        return facet_values

    def openface(self):
        """Returns a single-field dataset object for openface features"""
        openface_values = self.get_feature('openface')
        return openface_values

    def embeddings(self):
        """Returns a single-field dataset object for embeddings"""
        embeddings_values = self.get_feature('embeddings')
        return embeddings_values

    def words(self):
        """Returns a single-field dataset object for one-hot vectors of words"""
        words_values = self.get_feature('words')
        return words_values

    def phonemes(self):
        """Returns a single-field dataset object for one-hot vectors of phonemes"""
        phonemes_values = self.get_feature('phonemes')
        return phonemes_values

    def covarep(self):
        """Returns a single-field dataset object for covarep features"""
        covarep_values = self.get_feature('covarep')
        return covarep_values

    def opensmile(self):
        """Returns a single-field dataset object for opensmile features"""
        opensmile_values = self.get_feature('opensmile')
        return opensmile_values
    

class MOSI(Dataloader):
    """Dataloader for CMU-MOSI dataset"""
    def __init__(self):
        super(MOSI, self).__init__('MOSI')
