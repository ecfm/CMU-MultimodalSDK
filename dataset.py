#!/usr/bin/env python
"""
The file contains the class and methods for loading and aligning datasets
"""
import pickle

__author__ = "Prateek Vij"
__copyright__ = "Copyright 2017, Carnegie Mellon University"
__credits__ = ["Amir Zadeh", "Prateek Vij", "Soujanya Poria"]
__license__ = "GPL"
__version__ = "1.0.1"
__status__ = "Production"

class Dataset():
	"""Primary class for loading and aligning dataset"""

	def __init__(self, dataset_file, stored=False, timestamps='absolute'):
		"""Create a dictionary of features for the input dataset"""
		feature_dict = None
		if stored:
			self.dataset_pickle = dataset_file
			feature_dict = pickle.load(open(self.dataset_pickle))
			return feature_dict
		self.dataset_csv = dataset_file
		feature_dict = self.controller()
		return feature_dict

	def controller(self):
		"""
		Validates the dataset csv file and loads the features for the dataset
		from the files
		"""
		def validate_file(self):
			return True

		def load_features(self):
			return None

	def load_opensmile(self):
		return None

	def load_covarep(self):
		return None

	def p2fa(self):
		return None

	def load_w2v(self):
		return None

	def load_words(self):
		return None
		
	def load_openface(self):
		return None

	def load_facet(self):
		return None








