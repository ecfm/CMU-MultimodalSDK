## The datasets are currently unavailable for download until 01/18/2018. Please check the ACL 2018 workshop page: http://multicomp.cs.cmu.edu/acl2018multimodalchallenge/ ##

# CMU-MultimodalDataSDK

CMU-MultimodalDataSDK provides tools that facilitates simple and fast loading of well-known multimodal machine learning datasets such as CMU-MOSEI and CMU-MOSI. (The POM and ICT-MMMO datasets are coming soon!)

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

### 3.1 Installation

To start using this toolkit, simply clone this repository.

```bash
git clone git@github.com:A2Zadeh/CMU-MultimodalDataSDK.git
```

Then add the cloned folder to your `$PYTHONPATH` environment variable. For example you can do so by adding the following line (replace the path with your actual path of course) to your `~/.bashrc` file. 

```bash
export PYTHONPATH="/path/to/cloned/directory/CMU-MultimodalDataSDK:$PYTHONPATH"
```

Then it's all set.

### 3.2 Fetching Datasets ###

Now let's get started by an example for loading the CMU-MOSI dataset. We can choose from a variety of features for each dataset (for available features for each dataset, refer to section 3.8). For example, if we want to load the FACET features and word embeddings of CMU-MOSI, we do so by

```python
>>> import mmdata # import the multimodal data SDK
>>> mosi = mmdata.MOSI() # create a loader object for MOSI dataset
>>> mosi_facet = mosi.facet() # download the facet features for the first time and load it
>>> mosi_emb = mosi.embeddings() # download & load word embeddings
```

Simple as that. Now to explain the returned `mosi_facet` and `mosi_emb`. They are all provided as `Dataset` class objects (whose definition can be found in `mmdata/dataset.py`). These objects are designed so that different features can be merged into a larger `Dataset` easily, and most importantly, once you have a `Dataset` with multiple features, there's a class method for aligning the features' timestamps. We'll cover those details in the following sections.

### 3.3 Merging and Accessing Datasets

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

The data of any `Dataset` object can be accessed as if it is a nested dictionary. It has three levels, the first level of keys are the names of the features it contains, i.e 'embeddings', 'facet', 'covarep'.

```python
>>> mosi_facet.keys() # the first hierarchy of the nested dict is the feature names
['facet']
>>> mosi_facet_n_emb.keys()
['facet', 'embeddings']
```

The structure of a Dataset object is the same as a nested dictionary with 3 levels. You can access the data of a particular type of feature for a particular segment in a particular video by the following indexing: `feats[modality_name][video_id][segment_id]`. Here `modality_name` is just the name of the feature you want to access, e.g. 'facet', 'embeddings'. Video and segment IDs are strings that characterizes the video and segments in the dataset. While segment IDs are just strings of integers (e.g. '1', '2', '3', '16') indicating which segment it is within the video, video IDs usually doesn't have a pattern. But if you want to take a look at the video IDs, you can access them by looking at the keys of the second hierarchy of the nested dictionary.

```python
>>> vids = mosi_facet_n_emb['facet'].keys() # extract the list of all video ids
>>> vid = vids[0] # pick the first video
>>> segment_data = feats['facet'][vid]['3'] # access the facet data for the 3rd segment
```

Here, each segment data is a `list` of `tuple`s with the following format:

```python
segment_data = [
  (start_time_1, end_time_1, numpy.array([...])),
  (start_time_2, end_time_2, numpy.array([...])),
  (start_time_3, end_time_3, numpy.array([...])),
  				...			...
  (start_time_t, end_time_t, numpy.array([...])),
]
```

Each tuple contains a time slice indicated by start and end time and the corresponding feature vector for that time slice. And each segment has many such slices.

### 3.4 Feature Alignment

Next comes the most important functionality of the Dataset class: aligning the features based on a specified 'pivot' feature. It is a common problem that different features in multimodal machine learning are in different temporal frequencies, thus hard to combine. The alignment basically tries to make sure that the features are re-sampled so that they are all temporally aligned. For what exactly the alignment algorithm does, please refer to section 5. Here's the example code for aligning features according to the word embeddings.

```python
>>> aligned = mosi_facet_n_words.align('embeddings')
```

The resulting `aligned` is another nested dictionary that is of the same structure as the `feats` we've discussed before. After alignment, the pivot feature that is aligned to is dropped in this dictionary, so we usually use the feature `words` (which is one-hot vector features) as the pivot for word-level alignment. Note that `align` does not modify the original `Dataset`, so you'll have to keep the returned data in another variable.

### 3.5 Loading Train/Validation/Test Splits and Labels

In the CMU Multimodal Data SDK, train/validation/test splits are given as three Python `set`s of video IDs. Users can partition their obtained data according to the affiliations of their video IDs. The splits are obtained through the `.splits()` method. Such splits guarantees that segments from the same video will not be scattered across train/valid/test set.

```python
>>> train_ids = mosi.train()
>>> valid_ids = mosi.valid()
>>> test_ids = mosi.test()
```

Sentiment labels will be provided in nested dictionaries. The dictionary has two levels, such that one can access the labels by `labels[video_id][segment_id]`. labels are obtained through the following method:

```python
>>> labels = mosi.sentiments()
```

(For MOSEI dataset, the test set data as well as the sentiment labels will be released later.)

### 3.6 A Demo on Data Loading and Training Text-based LSTM

For a more comprehensive usage, you can refer to the demo `text_lstm.py` in the `CMU-MultimodalDataSDK/examples` directory. In order to run this demo, you'll need to install Keras and at least one of the backends (Tensorflow or Theano) it uses. This demo shows you how to download the embeddings and prepare the data to train an LSTM model.

### 3.7 Available Datasets and Features

Currently available datasets and multimodal features are:

|           | Visual          | Audio              | Textual                     |
| --------- | --------------- | ------------------ | --------------------------- |
| CMU-MOSI  | facet, openface | covarep, opensmile | words, embeddings, phonemes |
| CMU-MOSEI | facet           | covarep            | words, embeddings, phonemes |

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
