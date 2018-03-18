# News
Test set of CMU-MOSEI is now released!
CMU MultimodalDataSDK is introduced in ACL 2018 workshop on Computational Modeling of Human Multimodal Language: http://multicomp.cs.cmu.edu/acl2018multimodalchallenge/

# CMU-MultimodalDataSDK

CMU-MultimodalDataSDK provides tools that manage the **downloading, loading and preprocessing** of well-known multimodal machine learning datasets such as CMU-MOSEI and CMU-MOSI. (The POM and ICT-MMMO datasets are coming soon!)

## 1. CMU Multimodal Data SDK

CMU Multimodal Data SDK simplifies loading complex multimodal data. In multimodal datasets, data comes from multiple sources and is processed in different ways which makes loading this form of data very challenging. Often the researchers find themselves dedicating significant time and energy to loading the data before building models. CMU Multimodal Data SDK allows both users and developers to:

1. [user] donwload well-known multimodal datasets easily.
2. [user] load multimodal datasets very easily and align their modalities.
3. [developer] extend the SDK to your own data and publicizing your dataset. 

## 2. Citations

If you used this toolkit in your research, please cite the following publication:

```latex
@inproceedings{zadeh2018multi,
  title={Multi-attention Recurrent Network for Human Communication Comprehension},
  author={Zadeh, A and Liang, PP and Poria, S and Vij, P and Cambria, E and Morency, LP},
  booktitle={AAAI},
  year={2018}
}
```

Furthermore please cite the datasets used in your experiments.

## 3. Usage

In this section we outline how a user can utilize the CMU Multimodal Data SDK to easily load large-scale multimodal datasets. We demonstrate the usage through an example which involves three steps: 1) Fetching Datasets 2) Loading and Merging Data 3) Feature Alignment. 

### 3.1 Installation

To start using this toolkit, clone this repository.

```bash
git clone git@github.com:A2Zadeh/CMU-MultimodalDataSDK.git
```

Then add the cloned folder to your `$PYTHONPATH` environment variable. For example, you can do so by adding the following line (replace the path with your actual path of course) to your `~/.bashrc` file. 

```bash
export PYTHONPATH="/path/to/cloned/directory/CMU-MultimodalDataSDK:$PYTHONPATH"
```

Then this step is all set.

### 3.2 Fetching Datasets ###

Let's get started by an example for loading the CMU-MOSEI dataset. We can choose from a variety of features for each dataset (for available features for each dataset, refer to section 3.8). As an example, if we want to load FACET visual features and word embeddings of CMU-MOSEI, we do so by

```python
>>> from mmdata import Dataloader # import a Dataloader class from multimodal data SDK

>>> mosei = Dataloader('http://sorena.multicomp.cs.cmu.edu/downloads/MOSEI') # feed in the URL for the dataset. For URLs for all datasets, refer to section 3.7.

>>> mosei_facet = mosei.facet() # download & load facet visual feature

>>> mosei_emb = mosei.embeddings() # download & load word embeddings
```

Simple as that. 

Note that you always need to feed in the URL to the `Dataloader` object, in order to specify the dataset you want to load. If the dataset's files has been downloaded, it'll be loaded locally from your machine and won't be downloaded again.

Now to explain the returned `mosei_facet` and `mosei_emb` variables. They are all provided as `Dataset` class objects (definition can be found in `mmdata/dataset.py`). These objects are designed so that different features can be merged into a larger `Dataset` easily, and most importantly, once you have a `Dataset` with multiple features, there's a class method for aligning the features' based on timestamps. We'll cover those details in the following sections.

### 3.3 Merging and Accessing Datasets

So far we loaded the embeddings and facet features for CMU-MOSEI, next we want to merge these two unimodal `Dataset` instances into one multimodal `Dataset` instance to make them ready for the next step. And we also want to access the actual data inside. We'll go through these steps respectively.

 Here's an example of merging different features.

```python
>>> from mmdata import Dataloader, Dataset # we need the Dataset class for merging

>>> mosei = Dataloader('http://sorena.multicomp.cs.cmu.edu/downloads/MOSEI')

>>> mosei_facet = mosei.facet()

>>> mosei_emb = mosei.embeddings()

>>> mosei_facet_n_emb = Dataset.merge(mosei_facet, mosei_emb) # merge two Dataset object
```

The resulting `mosei_facet_n_words` is still a `Dataset` object, but now it contains 2 types of features.

The data of any `Dataset` object can be accessed as if it is a nested python dictionary. It has three levels. **The first level of keys are the names of the features it contains**, i.e 'embeddings', 'facet', 'covarep'. This may look a bit redundant for single-feature `Dataset`, but it is useful when you have multiple features in one `Dataset`.

```python
>>> mosei_facet.keys() # the first hierarchy of the nested dict is the feature names
['facet']

>>> mosei_facet_n_emb.keys()
['facet', 'embeddings']
```

From there, you can access the data of a particular type of feature for a particular segment in a particular video by the following indexing: `feats[modality_name][video_id][segment_id]`. Video and segment IDs are strings that characterizes the video and segments in the dataset. While segment IDs are **strings** of integers (e.g. '1', '2', '3', '16') indicating which segment it is within the video, video IDs usually doesn't have a pattern. If you want to take a look at the video IDs, you can access them by looking at the keys of the second hierarchy of the nested dictionary.

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

Each tuple contains a time slice indicated by start and end time and the corresponding feature vector for that time slice. For example for language modality such information could be (start of word utterance, end of word utternace, word embedding).

### 3.4 Feature Alignment

It is a common problem that different features in multimodal datasets are in different temporal frequencies, thus hard to combine. The alignment functionality of the SDK makes sure that the features are re-sampled such that they are all temporally aligned. For what exactly the alignment algorithm does, please refer to section 5. Here's the example code for aligning features according to the word embeddings.

```python
>>> aligned = mosei_facet_n_words.align('embeddings')
```

The resulting `aligned` is another nested dictionary that is of the same structure as the `feats` we've discussed before. Note that `align` does not modify the original `Dataset`, so you'll have to keep the returned data in another variable.

### 3.5 Loading Train/Validation/Test Splits and Labels

In the CMU Multimodal Data SDK, train/validation/test splits are given as three Python `set`s of video IDs. Users can partition their obtained data according to their video IDs. Such splits guarantees that segments from the same video will not be scattered across train/valid/test set.

```python
>>> train_ids = mosei.train()
>>> valid_ids = mosei.valid()
>>> test_ids = mosei.test()
```

Real-valued sentiment scores can be accessed through the following method:

```python
>>> labels = mosei.sentiments()
```

Sentiment labels will be provided also in nested dictionaries. The dictionary has two levels, and one can access the labels by `labels[video_id][segment_id]`. (For MOSEI dataset, the test set data as well as the test set sentiment labels will be released later.)

For some datasets (like MOSEI), we also have labels for emotion recognition. It can be load similarly.

```python
>>> emotions = mosei.emotions()
```

It is also a nested dictionary, and each emotion label is a vector that represents the intensity of different emotions.

### 3.6 Tutorials

For a more comprehensive usage, you can refer to the demo `early_fusion_lstm.py` in the `CMU-MultimodalDataSDK/examples` directory. In order to run this demo, you'll need to install Keras and at least one of the backends (Tensorflow or Theano) it uses. This demo shows you how to download the features on CMU-MOSI dataset and prepare the data to train an early-fusion LSTM model for multimodal sentiment analysis.

### 3.7 Available Datasets and Features

Currently available datasets and multimodal features are:

|           | Visual          | Audio              | Textual                     |
| --------- | --------------- | ------------------ | --------------------------- |
| CMU-MOSEI | facet           | covarep            | words, embeddings, phonemes |
| CMU-MOSI  | facet, openface | covarep, opensmile | words, embeddings, phonemes |
| IEMOCAP  | facet, openface | covarep, opensmile | words, embeddings, phonemes |
| MOUD  | facet, openface | covarep, opensmile | words, embeddings |
| MMMO  | facet, openface | covarep | words, embeddings, phonemes |
| POM  | facet, openface | covarep, opensmile | words, embeddings, phonemes |

Below are the URLs for each dataset:

| Dataset   | URL                                                   |
| --------- | ----------------------------------------              |
| CMU-MOSEI | http://sorena.multicomp.cs.cmu.edu/downloads/MOSEI    |
| CMU-MOSI  | http://sorena.multicomp.cs.cmu.edu/downloads/MOSI     |
| IEMOCAP   | http://sorena.multicomp.cs.cmu.edu/downloads/IEMOCAP  |
| MOUD      | http://sorena.multicomp.cs.cmu.edu/downloads/MOUD     |
| MMMO      | http://sorena.multicomp.cs.cmu.edu/downloads/MMMO     |
| POM  	    | http://sorena.multicomp.cs.cmu.edu/downloads/POM      | 

Below are the URLs for the Raw Files of the dataset:

| Dataset   | URL|
| --------- | ----------------------------------------              |
| CMU-MOSEI  | http://sorena.multicomp.cs.cmu.edu/downloads_raw/MOSEI     |
| CMU-MOSI  | http://sorena.multicomp.cs.cmu.edu/downloads_raw/MOSI     |
| IEMOCAP   | http://sorena.multicomp.cs.cmu.edu/downloads_raw/IEMOCAP  |
| MOUD      | http://sorena.multicomp.cs.cmu.edu/downloads_raw/MOUD     |
| MMMO      | http://sorena.multicomp.cs.cmu.edu/downloads_raw/MMMO     |
| POM  	    | http://sorena.multicomp.cs.cmu.edu/downloads_raw/POM      | 


If you are using any of these datasets, please cite the corresponding papers:

| Dataset   | Paper BibTeX                                              |
| --------- | ----------------------------------------              |
| CMU-MOSEI | Zadeh, Amir, et al. "Multimodal sentiment intensity analysis in videos: Facial gestures and verbal messages." IEEE Intelligent Systems 31.6 (2016): 82-88.|
| CMU-MOSI  | Zadeh, Amir, et al. "Multimodal sentiment intensity analysis in videos: Facial gestures and verbal messages." IEEE Intelligent Systems 31.6 (2016): 82-88.|
| IEMOCAP   | Busso, Carlos, et al. "IEMOCAP: Interactive emotional dyadic motion capture database." Language resources and evaluation 42.4 (2008): 335.|
| MOUD      | Pérez-Rosas, Verónica, Rada Mihalcea, and Louis-Philippe Morency. "Utterance-level multimodal sentiment analysis." Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Vol. 1. 2013.|
| MMMO      | Wöllmer, Martin, et al. "Youtube movie reviews: Sentiment analysis in an audio-visual context." IEEE Intelligent Systems 28.3 (2013): 46-53.|
| POM  	    | Park, Sunghyun, et al. "Computational analysis of persuasiveness in social multimedia: A novel dataset and multimodal prediction approach." Proceedings of the 16th International Conference on Multimodal Interaction. ACM, 2014.| 

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

The possible alignment strategies in this regard are:

** Weighted Averaging**

The two methods mentioned below use the concept of weighted averaging for alignment. The extent of overlap of segments of each modality with the reference modality segment is considered as the weight of each modality. An average is taken with these weights to align them to the reference. The methods are:

**1) Sub-sampling**

![Sub-Sampling](https://github.com/A2Zadeh/CMU-MultimodalDataSDK/blob/master/examples/Sub_Sampling.png)

In the subsampling method, given a large segment of the reference modality, we repeatedly fit as many multiple identical blocks of a modality segment to match the length of the reference. 

**2) Super-sampling** 

![Super-Sampling](https://github.com/A2Zadeh/CMU-MultimodalDataSDK/blob/master/examples/Super_Sampling.png)

In the supersampling method, a small piece of the reference modality is replicated to match the length of the larger modality segment.
