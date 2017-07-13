#!/usr/bin/env python
"""
The file contains the class and methods for loading and aligning datasets
"""
import pickle
import numpy as np
from scipy.io import loadmat

__author__ = "Prateek Vij"
__copyright__ = "Copyright 2017, Carnegie Mellon University"
__credits__ = ["Amir Zadeh", "Prateek Vij", "Soujanya Poria"]
__license__ = "GPL"
__version__ = "1.0.1"
__status__ = "Production"

class Dataset():
	"""Primary class for loading and aligning dataset"""

	def __init__(self, dataset_file, stored=False, timestamps='absolute'):
		"""
		Initialise the Dataset class. Support two loading mechanism - 
		from dataset files and from the pickle file, decided by the param
		stored.
		:param stored: True if loading from pickle, false if loading from 
					   dataset feature files. Default False
		:param dataset_file: Filepath to the file required to load dataset 
							 features. CSV or pickle file depending upon the
							 loading mechanism
		:timestamps: absolute or relative.
		"""
		self.feature_dict = None
		self.timestamps = timestamps
		self.stored = stored
		self.dataset_file = dataset_file

	def load(self):
		"""
		Loads feature dictionary for the input dataset
		:returns: Dictionary of features for the dataset with each modality 
		 as dictionary key
		"""

		# Load from the pickle file if stored is True
		if self.stored:
			self.dataset_pickle = self.dataset_file
			self.feature_dict = pickle.load(open(self.dataset_pickle))
			return self.feature_dict

		# Load the feature dictionary from the dataset files
		self.dataset_csv = self.dataset_file
		self.feature_dict = self.controller()
		return self.feature_dict

	def controller(self):
		"""
		Validates the dataset csv file and loads the features for the dataset
		from its feature files
		"""

		def validate_file(self):
			return True

		def load_features(self):
			return None

		return None

	def load_opensmile(self, filepath, start, end, timestamps='absolute', level='s'):
		"""
		Load OpenSmile Features from the file corresponding to the param
		filepath
		:param start: Start time of the segment
		:param end: End time of the segment
		:param filepath: Path to the opensmile feature files
		:param level: 's' if the file contains features only for the segment,
					  i.e. interval (start, end), 'v' if for the entire video 
		:param timestamps: relative or absolute
		:returns: List of tuples (feat_start, feat_end, feat_value)
				  corresponding to the features in the interval.
		Note: Opensmile support features for entire segment or video only and 
			  will return None if level is 'v' and start time is 
		"""
		features = []
		start_time, end_time = start, end
		if timestamps == 'relative':
			start_time = 0.0
			end_time = end - start

		if level == 's' or start == 0.0:
			feats = open(filepath).readlines()[-1].strip().split(',')[1:]
			feats = [float(feat_val) for feat_val in feats]
			feat_val = np.asarray(feats, dtype=np.float32)
			features.append((start_time, end_time, feat_val))
		else:
			print "Opensmile support features for the entire segment"
			return None
		return features

	def load_covarep(self, filepath, start, end, timestamps='absolute', level='s'):
		"""
		Load COVAREP Features from the file corresponding to the param 
		filepath
		:param start: Start time of the segment
		:param end: End time of the segment
		:param filepath: Path to the opensmile feature files
		:param level: 's' if the file contains features only for the segment,
					  i.e. interval (start, end), 'v' if for the entire video 
		:param timestamps: relative or absolute
		:returns: List of tuples (feat_start, feat_end, feat_value)
				  corresponding to the features in the interval.
		"""
		features = []
		time_period = 0.01
		f_content = loadmat(filepath)
		feats = f_content['features']
		start_time, end_time = start, end
		if timestamps == "relative":
			start_time, end_time = 0.0, end - start

		if level == 's':
			feat_start = start_time
			for feat in feats:
				feat_end = feat_start + time_period
				feat_val = np.asarray(feat)
				features.append((feat_start, feat_end, feat_val))
				feat_start += time_period
		else:
			feat_count = feats.shape[0]
			start_index = min((start/time_period), feat_count)
			end_index = min((end/time_period), feat_count)
			feat_start = start_time
			for feat in feats[start_index:end_index]:
				feat_end = feat_start + time_period
				feat_val = np.asarray(feat)
				features.append((feat_start, feat_end, feat_val))
				feat_start += time_period
		return features

	def load_p2fa_p(self, filepath, start, end, timestamps='absolute', level='v'):
		"""
		Load P2FA phonemes as Features from the file corresponding to the 
		param filepath
		:param start: Start time of the segment
		:param end: End time of the segment
		:param filepath: Path to the opensmile feature files
		:param level: 's' if the file contains features only for the segment,
					  i.e. interval (start, end), 'v' if for the entire video 
		:param timestamps: relative or absolute
		:returns: List of tuples (feat_start, feat_end, feat_value)
				  corresponding to the features in the interval.
		"""
		features = []
		start_time, end_time = start, end
		if timestamps == "relative":
			start_time, end_time = 0.0, end - start

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
					feat_start = float(line) + start_time

				elif i%3 == 1:
					feat_end = float(line) + start_time

				else:
					if line.startswith('"') and line.endswith('"'):
						feat_val = line[1:-1]
						features.append((feat_start, feat_end, feat_val))
					else:
						print "File format error at line number ",str(i)
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

							feat_val = line[1:-1]
							feat_start = feat_start - start + start_time
							feat_end = feat_end - start + start_time
							features.append((feat_start, feat_end, feat_val))
					else:
						print "File format error at line number ",str(i)

		return features

	def load_embeddings(self, filepath, start, end, timestamps='absolute', level='v'):
		"""
		Load Word Embeddings from the file corresponding to the param 
		filepath
		:param start: Start time of the segment
		:param end: End time of the segment
		:param filepath: Path to the opensmile feature files
		:param level: 's' if the file contains features only for the segment,
					  i.e. interval (start, end), 'v' if for the entire video 
		:param timestamps: relative or absolute
		:returns: List of tuples (feat_start, feat_end, feat_value)
				  corresponding to the features in the interval.
		"""
		features = []
		start_time, end_time = start, end
		if timestamps == "relative":
			start_time, end_time = 0.0, end - start

		if level == 's':
			with open(filepath, 'r') as f_handle:
				for line in f_handle.readlines():
					line = line.strip()
					if not line:
						break
					feat_start = float(line.split(",")[0]) + start_time
					feat_end = float(line.split(",")[1]) + start_time
					feat_val = [float(val) for val in line.split(",")[3:]]
					feat_val = np.asarray(feat_val)
					features.append((feat_start, feat_end, feat_val))
		else:
			with open(filepath,'r') as f_handle:
				for line in f_handle.readlines():
					line = line.strip()
					if not line:
						break
					feat_start = float(line.split(",")[0])
					feat_end = float(line.split(",")[1])
					if ((feat_start <= start and feat_end > end) 
					   or (feat_start >= start and feat_end < end)
					   or (feat_start <= start 
						  and start-feat_start < feat_time/2)
					   or (feat_start >= start
						  and end - feat_start > feat_time/2)):
						
						feat_start = feat_start - start + start_time
						feat_end = feat_end - start + start_time
						feat_val = [float(val) for val in line.split(",")[3:]]
						feat_val = np.asarray(feat_val)
						features.append((feat_start, feat_end, feat_val))
		return features

	def load_p2fa_w(self, filepath, start, end, timestamps='absolute', level='v'):
		"""
		Load P2FA words as features from the file corresponding to the param 
		filepath
		:param start: Start time of the segment
		:param end: End time of the segment
		:param filepath: Path to the opensmile feature files
		:param level: 's' if the file contains features only for the segment,
					  i.e. interval (start, end), 'v' if for the entire video 
		:param timestamps: relative or absolute
		:returns: List of tuples (feat_start, feat_end, feat_value)
				  corresponding to the features in the interval.
		"""
		features = []
		start_time, end_time = start, end
		if timestamps == "relative":
			start_time, end_time = 0.0, end - start

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
					feat_start = float(line) + start_time

				elif i%3 == 1:
					feat_end = float(line) + start_time

				else:
					if line.startswith('"') and line.endswith('"'):
						feat_val = line[1:-1]
						features.append((feat_start, feat_end, feat_val))
					else:
						print "File format error at line number ",str(i)
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

							feat_val = line[1:-1]
							feat_start = feat_start - start + start_time
							feat_end = feat_end - start + start_time
							features.append((feat_start, feat_end, feat_val))
					else:
						print "File format error at line number ",str(i)

		return features

	def load_openface(self, filepath, start, end, timestamps='absolute', level='v'):
		"""
		Load OpenFace features from the file corresponding to the param 
		filepath
		:param start: Start time of the segment
		:param end: End time of the segment
		:param filepath: Path to the opensmile feature files
		:param level: 's' if the file contains features only for the segment,
					  i.e. interval (start, end), 'v' if for the entire video 
		:param timestamps: relative or absolute
		:returns: List of tuples (feat_start, feat_end, feat_value)
				  corresponding to the features in the interval.
		"""
		features = []
		time_period = 0.0333333

		start_time, end_time = start, end
		if timestamps == "relative":
			start_time, end_time = 0.0, end - start

		if level == 's':
			with open(filepath, 'r') as f_handle:
				for line in f_handle.readlines()[1:]:
					line = line.strip()
					if not line:
						break
					feat_start = float(line.split(",")[0]) + start_time
					feat_end = feat_start + time_period
					feat_val = [float(val) for val in line.split(",")[1:]]
					feat_val = np.asarray(feat_val, dtype=np.float32)
					features.append((feat_start, feat_end, feat_val))

		else:
			with open(filepath, 'r') as f_handle:
				for line in f_handle.readlines()[1:]:
					line = line.strip()
					if not line:
						break
					feat_start = float(line.split(",")[1])

					if (feat_start >= start and feat_start < end):
						# To adjust the timestamps
						feat_start = feat_start - start + start_time
						feat_end = feat_start + time_period
						feat_val = [float(val) for val in line.split(",")[2:]]
						feat_val = np.asarray(feat_val, dtype=np.float32)
						features.append((feat_start, feat_end, feat_val))
		return features

	def load_facet(self, filepath, start, end, timestamps='absolute', level='v'):
		"""
		Load FACET features from the file corresponding to the param 
		filepath
		:param start: Start time of the segment
		:param end: End time of the segment
		:param filepath: Path to the opensmile feature files
		:param level: 's' if the file contains features only for the segment,
					  i.e. interval (start, end), 'v' if for the entire video 
		:param timestamps: relative or absolute
		:returns: List of tuples (feat_start, feat_end, feat_value)
				  corresponding to the features in the interval.
		"""
		features = []
		time_period = 0.03333

		start_time, end_time = start, end
		if timestamps == "relative":
			start_time, end_time = 0.0, end - start

		if level == 's':
			with open(filepath, 'r') as f_handle:
				for line in f_handle.readlines()[1:]:
					line = line.strip()
					if not line:
						break
					feat_start = float(line.split(",")[0]) + start_time
					feat_end = feat_start + time_period
					feat_val = [float(val) for val in line.split(",")[1:]]
					feat_val = np.asarray(feat_val, dtype=np.float32)
					features.append((feat_start, feat_end, feat_val))

		else:
			with open(filepath, 'r') as f_handle:
				for line in f_handle.readlines()[1:]:
					line = line.strip()
					if not line:
						break
					feat_start = float(line.split(",")[0])

					if (feat_start >= start and feat_start < end):
						# To adjust the timestamps
						feat_start = feat_start - start + start_time
						feat_end = feat_start + time_period
						feat_val = [float(val) for val in line.split(",")[1:]]
						feat_val = np.asarray(feat_val, dtype=np.float32)
						features.append((feat_start, feat_end, feat_val))
		return features
