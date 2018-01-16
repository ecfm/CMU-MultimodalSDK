## The datasets are currently unavailable for download until 01/18/2018. Please check the ACL 2018 workshop page: http://multicomp.cs.cmu.edu/acl2018multimodalchallenge/ ##

# CMU-MultimodalDataSDK

CMU-MultimodalDataSDK provides tools that facilitates simple and fast loading of well-known multimodal machine learning datasets such as CMU-MOSEI, CMU-MOSI, POM, and ICT-MMMO. 

## 1. CMU Multimodal Data SDK

CMU Multimodal Data SDK simplifies loading complex multimodal data. Often cases in different multimodal datasets, data comes from multiple sources and is processed in different ways which makes loading this form of data very challenging. Often the researchers find themselves dedicating significant time and energy to loading the data before building models. CMU Multimodal Data SDK allows both users and developer to:

1. [user] load multimodal datasets very easily and align their modalities.
2. [user] donwload well-known multimodal datasets easily.
3. [developer] extend the SDK to your own data and publicizing your dataset. 

## 2. Citations

If you used this toolkit in your research, please cite the following publication:

```latex
@inproceedings{tensoremnlp17,
title={Tensor Fusion Network for Multimodal Sentiment Analysis},
author={Zadeh, Amir and Chen, Minghai and Poria, Soujanya and Cambria, Erik and Morency, Louis-Philippe},
booktitle={Empirical Methods in Natural Language Processing, EMNLP},
year={2017}
}
```

## 3. Usage

In this section we outline how a user can utilize the CMU Multimodal Data SDK to easily load large-scale multimodal datasets. We demonstrate the usage through an example which involves three steps: 1) Fetching Datasets 2) Loading and Merging Data 3) Feature Alignment. 

### 3.1 Fetching Datasets

To start using this toolkit, simply clone this repository.

```bash
git clone git@github.com:A2Zadeh/CMU-MultimodalDataSDK.git
```

Then add the cloned folder to your `$PYTHONPATH` environment variable. For example you can do so by adding the following line (replace the path with your actual path of course) to your `~/.bashrc` file. 

```bash
export PYTHONPATH="/path/to/cloned/directory/CMU-MultimodalDataSDK:$PYTHONPATH"
```

Then it's all set.

Now let's get started by an example loading the CMU-MOSI dataset. We provide several different multi-modal features for each of our datasets. For example, if we want to load the FACET features and word embeddings of CMU-MOSI, we do so by

```python
>>> import mmdata
>>> mosi = mmdata.MOSI() # create a loader object for MOSI dataset
>>> mosi_facet = mosi.facet() # download the facet features for the first time and load it
>>> mosi_emb = mosi.embeddings() # download & load word embeddings
```

Simple as that. Now to explain the returned `mosi_facet` and `mosi_emb`. All the features are provided as `Dataset` class objects (whose definition can be found in `mmdata/dataset.py`). They are designed so that different features can be merged into a larger `Dataset`, and most importantly, once you have a `Dataset` with multiple features, there's a class method for aligning the features' timestamps. We'll cover those details in the following sections.

### 3.2 Merging and Accessing Datasets

Now that we have loaded the embeddings and facet features for CMU-MOSI, we may want to merge these two uni-feature `Dataset` into one `Dataset` to make them ready for the next step. And we also want to access the actual data inside. We'll go through the respectively.

 Here's an example of merging different features.

```python
>>> import mmdata
>>> from mmdata import Dataset # we need the Dataset class for merging
>>> mosi = mmdata.MOSI()
>>> mosi_facet = mosi.facet()
>>> mosi_emb = mosi.embeddings()
>>> mosi_facet_n_emb = Dataset.merge(mosi_facet, mosi_emb) # merge the two Datasets
```

The resulting `mosi_facet_n_words` is still a `Dataset` object, but now it contains 2 types of features.

The data of any `Dataset` object can be accessed from its `feature_dict` attribute. As the name suggests, the actual features are stored in a nested dictionaries structure.

```python
>>> feats = mosi_facet_n_words.feature_dict # the actual data now in the feats variable
>>> feats.keys() # the first hierarchy of the nested dict is the feature types
['facet', 'embeddings']
```

The structure of `feats` is a nested dictionary with 3 levels. You can access the data of a particular type of feature for a particular segment in a particular video by the following indexing: `feats[modality_name][video_id][segment_id]`. Here `modality_name` is just the name of the feature you want to access, e.g. 'facet', 'embeddings'. Video and segment IDs are strings that characterizes the video and segments in the dataset. While segment IDs are just strings of integers (e.g. '1', '2', '3', '16') indicating which segment it is within the video, video IDs usually doesn't have a pattern. But if you want to take a look at the video IDs, you can access them by looking at the keys of the second hierarchy of the nested dictionary.

```python
>>> vids = feats['facet'].keys() # extract the list of all video ids
>>> vid = vids[0] # pick the first video
>>> segment_data = feats['facet'][vid]['3'] # access the facet data for the 3rd segment
```

As a side note, `Dataset` that has only one type of features (usually obtained through methods mentioned in section 3.1) will have only one key in the first level, with the key being the same as the method that is used to load the feature. For a more detailed explanation for the feature dictionaries, refer to section 4.

### 3.3 Feature Alignment

Next comes the most important functionality of the Dataset class: aligning the features based on a specified 'pivot' feature. For what exactly the alignment algorithm does, please refer to section 5. Here's an example for aligning features according to the word embeddings.

```python
>>> aligned = mosi_facet_n_words.align('embeddings')
```

The resulting `aligned` is another nested dictionary that is of the same structure as the `feats` we've discussed before. Note that the pivot feature that is aligned to is dropped in this dictionary. Note that `align` does not modify the original `Dataset`, so its `feature_dict` will still be the original, unaligned version.

### 3.4 A Demo on Loading the Full Dataset and Train Text-based LSTM

For a more comprehensive usage, you can refer to the demo `text_lstm.py` in the `CMU-MultimodalDataSDK/` directory. In order to run this demo, you'll need to install Keras and at least one of the backends (Tensorflow or Theano) it uses. This demo shows you how to download the full dataset and Links for Features

### 3.5 Full Feature Set:

For the full feature set, there are two options:

1. Download the entire dataset along with Raw video, audio and text files with processed features also.
  Link: http://sorena.multicomp.cs.cmu.edu/downloads/mosi/full/MOSI.tar.gz

When the full dataset is downloaded, the full feature set can be loaded onto the dictionary simply by calling the load() function as mentioned below. All code files are present in the lib/ directory.

```python
from mmdata.dataset import Dataset	

csv_fpath = "../configs/CMU_MOSI_all.csv"

# Code for loading
d = Dataset(csv_fpath)
features = d.load() # this gives you the feature_dict
```
This loads the features into a dictionary for use. The structure of the dictionary is explained below under the section "Dictionary Structure".

2. Download a pickled dictionary file containing the unaligned feature set for direct use:
  Link: http://sorena.multicomp.cs.cmu.edu/downloads/mosi/full/MOSI_before_align.pkl

When the dictionary is downloaded, it bypasses the load step mentioned above and can be directly used as the feature dictionary (the structure of the dictionary is mentioned below under the section "Dictionary Structure")

It is worthy to note here that this dictionary is the unaligned dictionary. This was done to provide more freedom to the user to choose their own plan of action to perform on the unaligned loaded dictionary. The section below on alignment strategies explains the simple method to align the features and return the aligned dictionary of features for use. 

### 3.6 Individual Features:

Input the name of the feature you want to download in the link below to obtain the tarball.

http://sorena.multicomp.cs.cmu.edu/downloads/mosi/separate/ [your-feature-name-here].tar.gz

Visual Features:

1. FACET
2. OpenFace

Audio features:

1. COVAREP
2. OpenSmile

Text Features:

1. embeddings
2. words
3. phonemes

Labels:

1. labels

## 4. Dictionary Structure

As also mentioned above, most of the times, apart from the Raw data, we also provide a dictionary loaded with the segmented features of each segment in each video in each modality.

The dictionary of loaded features contains the following structure:

```
Features = { modality_0: {
                            video_id_0: {
                                    segment_0: [Feature_1, Feature_2, Feature_3,......],
                                    segment_1: [Feature_1, Feature_2, Feature_3,......],
                                    segment_2: [Feature_1, Feature_2, Feature_3,......],
                                    ....
                                  }, 

                            video_id_1: {
                                    segment_0: [Feature_1, Feature_2, Feature_3,......],
                                    segment_1: [Feature_1, Feature_2, Feature_3,......],
                                    segment_2: [Feature_1, Feature_2, Feature_3,......],
                                    ....
                                  },
                            .
                            .
                            .
                            . 
                          },

      	    modality_1: {	
			   ...
			}, 
            .
            .
            .
            .	 
          }
```

## 5. Alignment Strategies

Alignment of modalities form an important component in Multimodal Machine Learning. To completely leverage the power of the modalities combined together, there should be a uniform convention or reference point over which each modality is aligned to help capture them together. Here, we take any one of the modalities as our reference point with which other modalities are aligned.

Given a reference modality, our objective is to match as accurately as possible the exact time frames of occurrence of the same event among all other modalities. 

The beginning and end of the reference modality is denoted by the variables start_interval and end_interval respectively. The beginning and end of the other modality that is to be aligned with the reference is denoted by feat_start and feat_end respectively 

There are three possible alignment strategies in this regard:

**1) Weighted Averaging**

In the weighted averaging method, the extent of overlap of segments of each modality with the reference modality segment is considered as the weight of each modality. An average is taken with these weights to align them to the reference.

**2) Subsampling**

In the subsampling method, given a large segment of the reference modality, we repeatedly fit as many multiple identical blocks of a modality segment to match the length of the reference. 

**3) Supersampling** 

In the supersampling method, a small piece of the reference modality is replicated to match the length of the larger modality segment.

The given dictionary (it is also present as a downloadable file in the repository) when loaded with the data can be aligned by simply calling the align() function. A sample code snippet is as shown below:

```python
# Assuming the dictionary is stored in the variable "mosi_dict"

	print mosi_dict.modalities	# shows all modalities loaded in the dictionary and the modality corresponding to its modality codes (eg: {word_embeddings: modality_0, phonemes: modality_1, etc..} )

	mosi_dict_aligned = mosi_dict.align('modality_0') # Function to align all modalities with respect to modality_0. 

# In the dictionary, each key is only the modality_code and not the name of the modality itself. Hence only the modality_code needs to be passed to the function.	

```
