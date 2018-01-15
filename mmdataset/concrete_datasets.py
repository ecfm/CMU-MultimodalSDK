import os
from .dataset import Dataset
from cPickle import load


class MOSI(Dataset):
    """Dataset object for CMU-MOSI dataset"""

    def __init__(self):
        super(MOSI, self).__init__()
