import os
from dataset import Dataset
from cPickle import load
from utils import download

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
                feature_values = load(fp)
            except:
                print "The previously downloaded dataset is compromised, downloading a new copy..."
                dowloaded = download(self.dataset, feature, self.location)
                if not downloaded:
                    return None
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

    def sentiments(self):
        """Returns a nested dictionary that stores the sentiment values"""
        sentiments_values = self.get_feature('sentiments')
        return sentiments_values

    def emotions(self):
        """Returns a nested dictionary that stores the emotion distributions"""
        emotions_values = self.get_feature('emotions')
        return emotions_values

    def train(self):
        """Returns three sets of video ids: train, dev, test"""
        train_ids = self.get_feature('train')
        return train_ids

    def valid(self):
        """Returns three sets of video ids: train, dev, test"""
        valid_ids = self.get_feature('valid')
        return valid_ids

    def test(self):
        """Returns three sets of video ids: train, dev, test"""
        test_ids = self.get_feature('test')
        return test_ids

class MOSI(Dataloader):
    """Dataloader for CMU-MOSI dataset"""
    def __init__(self):
        super(MOSI, self).__init__('http://sorena.multicomp.cs.cmu.edu/downloads/MOSI')
        print "This API will be deprecated in the future versions. Please check the Github page for the current API"


class MOSEI(Dataloader):
    """Dataloader for CMU-MOSEI dataset"""
    def __init__(self):
        super(MOSEI, self).__init__('http://sorena.multicomp.cs.cmu.edu/downloads/MOSEI')
        print "This API will be deprecated in the future versions. Please check the Github page for the current API"
