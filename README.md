# CMU-MultimodalDataSDK
MultimodalSDK provides tools to easily apply machine learning algorithms on well-known affective computing datasets such as CMU-MOSI, CMU-MOSI2, POM, and ICT-MMMO. 

## CMU Multimodal Data SDK

CMU Multimodal Data SDK simplifies loading complex multimodal data. Often cases in many different multimodal datasets, data comes from multiple sources and is processed in different ways. The difference in the nature of the data and the difference in the processing makes loading this form of data very challenging. Often the researchers find themselves dedicating significant time and energy to loading the data before building models. CMU Multimodal Data SDK allows you to load and align multimodal datasets very easily. These datasets normally come in the form of video segments with labels. This SDK comes with functionalities already implemented for a variety of processed outputs. Furthermore it is easy to add functionalities to load new form of outputs to the SDK. In its core the following outputs are already supported:
1. Loading time-distributed data coming in the form of start_time, end_time, feature 1, feature 2, ...
2. JSON file for alignment between words and phonemes with audio

## Alignment Strategies:

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

```
# Assuming the dictionary is stored in the variable "mosi_dict"

	print mosi_dict.modalities	# shows all modalities loaded in the dictionary and the modality corresponding to its modality codes (eg: {word_embeddings: modality_0, phonemes: modality_1, etc..} )

	mosi_dict_aligned = mosi_dict.align('modality_0') # Function to align all modalities with respect to modality_0. 

# In the dictionary, each key is only the modality_code and not the name of the modality itself. So please be careful to pass only the modality_code to the align() function.	

```

## Dictionary Structure:

As also mentioned above, most of the times, apart from the Raw data, we also provide a dictionary loaded with the segmented features of each segment in each video in each modality. This can be downloaded as a file: <file_name>

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
## Specify Features You Want To Load ##

The CMU Multimodal Data SDK uses CSV files to store queries for features. Typically you can specify everything you need in one CSV per dataset.

These CSV files should have $n+4$ columns, with $n$ columns corresponding to the different types of features you want to load, and an additional $4$ columns that stores information as to for which video segment you want these features. Take for example you want to load FACET and COVAREP features, then your columns should be:

| video_id | segment | start | end  | facet | covarep |
| -------- | ------- | ----- | ---- | ----- | ------- |
| ...      | ...     | ...   | ...  | ...   | ...     |

The first 4 columns are for specifying the video segment. They're the video ID (basically the file name of the video), the segment ID, the start time of the segment and the end time. For videos that doesn't come in segments, segment ID is always set to 1. Then the trailing columns are for specific features. Please note that if you're using standard features shipped together with the dataset, then please remember to use a standard alias of the features (listed below) for the columns, since those strings will be used to determine which loading subprocess to go through.

List of possible off-the-shelf features and their standard aliases:

```
One-hot vectors of words: words
Glove word embeddings: embeddings
Phonemes: phonemes
OpenFace: openface
FACET: facet
Opensmile: opensmile
COVAREP: covarep
```

Note that a very important feature of the CMU Multimodal Data SDK is that it supports loading both features stored at segment level and video level, but you always HAVE TO explicitly specify that. Continued from the previous example where you load FACET which is stored at video level, and COVAREP which is stored segment level, the first two rows of your CSV should be:

| video_id | segment | start | end  | facet | covarep |
| -------- | ------- | ----- | ---- | ----- | ------- |
|          |         |       |      | v     | s       |

Notice how for the second row the first 4 columns are left as blank, and for each of the feature columns, there is a flag "v" or "s", indicating whether this feature should be loaded at video level or segment level. So this structure of 2-row header determines which features to use and on what level they are loaded. Below the 2-row header, we'll store actual queries. Let's see an example.

For loading FACET and COVAREP for a video segment with video ID `testvideo123` and segment ID 3, starting from 23.32s to 32.12s in the original video, we use the following CSV:

|    video_id    | segment | start |  end  |       facet       |       covarep       |
| :------------: | :-----: | :---: | :---: | :---------------: | :-----------------: |
|                |         |       |       |         v         |          s          |
| `testvideo123` |    3    | 23.32 | 32.12 | \<path/to/facet\> | \<path/to/covarep\> |

The feature columns contains paths to the files that stores the feature. Paths to features always follows the following format: `/dataset/processed/features/[modality]/[feature alias]/[video_id]_[segment_id].[suffix]`. The ones with square brackets should be replaced with the actual modality, feature, video and segment ID, etc., you use. **Note that the segment ID is omitted for video level feature files.**

Following this pattern, the FACET path for `testvideo123` will be:

`/dataset/processed/features/visual/testvideo123.csv`

and the COVAREP path is:

`/dataset/processed/features/audio/testvideo123_3.mat`

since it is on segment level.  

 
## Tutorial ##
A short tutorial on how to develop machine learning models using CMU-MultimodalDataSDK and Keras is available as `text_lstm.py`. You can simply use `python text_lstm.py` to train a unimodal text-based sentiment analysis model on MOSI. Feel free to explore the code.