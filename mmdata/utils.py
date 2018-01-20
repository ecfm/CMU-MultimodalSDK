#!/usr/bin/env python
"""
The file contains the class and methods for loading and aligning datasets
"""
import numpy as np
import urllib2
import sys
import os
from subprocess import call

__author__ = "Prateek Vij"
__copyright__ = "Copyright 2017, Carnegie Mellon University"
__credits__ = ["Amir Zadeh", "Prateek Vij", "Soujanya Poria"]
__license__ = "GPL"
__version__ = "1.0.1"
__status__ = "Production"

p2fa_phonemes = [ "EH2", "K", "S", "L", "AH0", "M", "EY1", "SH", "N", "P", "OY2", "T", "OW1", "Z", "W", "D", "AH1", "B", "EH1", "V", "IH1", "AA1", "R", "AY1", "ER0", "AE1", "AE2", "AO1", "NG", "G", "IH0", "TH", "IY2", "F", "DH", "IY1", "HH", "UH1", "IY0", "OY1", "OW2", "CH", "UW1", "IH2", "EH0", "AO2", "AA0", "AA2", "OW0", "EY0", "AE0", "AW2", "AW1", "EY2", "UW0", "AH2", "UW2", "AO0", "JH", "Y", "ZH", "AY2", "ER1", "UH2", "AY0", "ER2", "OY0", "UH0", "AW0", "br", "cg", "lg", "ls", "ns", "sil", "sp" ]

def phoneme_index(phoneme):
    return p2fa_phonemes.index(phoneme)

def phoneme_hotkey_enc(phoneme):
    index = phoneme_index(phoneme)
    enc = np.zeros(len(p2fa_phonemes))
    enc[index] = 1
    return enc

def download(dataset, feature, dest):
    call(['mkdir', '-p', dest])
    url = dataset + '/' + feature + '.pkl'
    file_path = os.path.join(dest, feature + '.pkl')
    print file_path

    try:
        u = urllib2.urlopen(url)
    except urllib2.HTTPError:
        print "The requested data is not available for {} dataset.".format(dataset)
        return False
    with open(file_path, 'wb') as f:
        meta = u.info()
        file_size = int(meta.getheaders("Content-Length")[0])
        print "Downloading: {}, size: {}".format(' '.join([dataset, feature]), file_size)

        file_size_dl = 0
        block_sz = 8192
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break
            file_size_dl += len(buffer)
            f.write(buffer)
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] [%3.2f%%]" % ('='*int((file_size_dl * 100. / file_size)/5), file_size_dl * 100. / file_size))
            sys.stdout.flush()
    sys.stdout.write('\n')
    return True
