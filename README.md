

# CMU-Multimodal SDK Version 1.0.2 (mmsdk)

CMU-Multimodal SDK provides tools to easily load well-known multimodal datasets and rapidly build neural multimodal deep models. Hence the SDK comprises of two modules: 1) mmdatasdk: module for downloading and procesing multimodal datasets using computational sequences. 2) mmmodelsdk: tools to utilize complex neural models as well as layers for building new models (released Jan 1 2019). 

# News

CMU-MOSI COVAREP has been added. Please pull the SDK to be able to use it. 

Some examples are released to clarify confusions about downloading and aligning datasets.

**Raw data now available for download outside SDK**. You can download the raw data as well. I strongly recommend sticking to SDK for running machine learning studies. If you want to extract your own features you can create computational sequences and share them with us and others. All raw data can be downloaded from http://immortal.multicomp.cs.cmu.edu/raw_datasets/. 

Update: POM Dataset Added (version 1.0.3 announced). As the next step, we will add more tutorials and add functionalities for passive alignment. We will also release the raw data (currently too big so we are looking for solution on where to put them). 

**To see what our next steps are for the SDK please look at next_steps.md**


## CMU Multimodal Data SDK (mmdatasdk)

CMU-Multimodal Data SDK simplifies downloading nad loading multimodal datasets. The module mmdatasdk treats each multimodal dataset as a combination of **computational sequences**. Each computational sequence contains information from one modality in a heirarchical format, defined in the continuation of this section. Computational sequences are self-contained and independent; they can be used to train models in isolation. They can be downloaded, shared and registered with our trust servers. This allows the community to share data and recreate results in a more elegant way using computational sequence intrgrity checks. Furthermore, this integrity check allows users to download the correct computational sequences. 

Each computational sequence is a heirarchical data strcuture which contains two key elements 1) "data" is a heirarchy of features in the computational sequence categorized based on unique multimodal source identifier (for example video id). Each multimodal source has two matrices associated with it: features and intervals. Features denote the computational descriptors and intervals denote their associated timestamp. Both features and intervals are numpy 2d arrays. 2) "metadata": contains information about the computational sequence including integrity and version information. The computational sequences are stored as hdf5 objects on hard disk with ".csd" extension (computational sequential data). Both the data and metadata are stored under "root name" (root of the heirarchy)

A dataset is defined as a dictionary of multiple computational sequences. Entire datasets can be shared using recipes as opposed to old-fashioned dropbox links or ftp servers. Computational sequences are downloaded one by one and their individual integrity is checked to make sure they are the ones users wanted to share. Users can register their extracted features with our trust server to use this feature. They can also request storage of their features on our servers 


## Installation

The first step is to download the SDK:

```bash
git clone git@github.com:A2Zadeh/CMU-MultimodalSDK.git
```

Then add the cloned folder to your `$PYTHONPATH` environment variable. For example, you can do so by adding the following line (replace the path with your actual path of course) to your `~/.bashrc` file. 

```bash
export PYTHONPATH="/path/to/cloned/directory/CMU-MultimodalSDK:$PYTHONPATH"
```

Make sure the following python packages are installed: h5py, validators, tqdm. The setup.py will install them for you. You can also manually install them using pip by:

```bash
pip install h5py validators tqdm numpy argparse
```

## Usage

The first step in most machine learning tasks is to acquire the data. We will work with CMU-MOSI for this readme. 

```python
>>> from mmsdk import mmdatasdk
```

Now that mmdatasdk is loaded you can proceed to fetch a dataset. The datasets are a set of computational sequences, where each computational sequence hosts the information from a modality or a view of a modality. For example a computational sequence could be the word vectors and another computational sequence could be phoneme 1-hot vectors. 

If you are using a standard dataset, you can find the list of them in the mmdatasdk/dataset/standard_datasets. We use CMU-MOSI for now. We will work with highlevel features (glove embeddings, facet facial expressions, covarep acoustic features, etc)

```python
>>> from mmsdk import mmdatasdk
>>> cmumosi_highlevel=mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.highlevel,'cmumosi/')
```

This will download the data using the links provided in *mmdatasdk.cmu_mosi.highlevel* dictionary (mappings between computational sequence keys and their respective download link) and put them in the *cmumosi/* folder. 

The data that gets downloaded comes in different frequencies, however, they computational sequence keys will always be the same. For example if video v0 exists in glove embeddings, then v0 should exist in other computational sequences as well. The data with different frequency is applicable for machine learning tasks, however, sometimes the data needs to be aligned. The next stage is to align the data according to a modality. For example we would like to align all computational sequences according to the labels of a dataset. First, we fetch the opinion segment labels computational sequence for CMU-MOSI. 

```python
>>> cmumosi_highlevel.add_computational_sequence(mmdatasdk.cmu_mosi.labels,'cmumosi/')
```

Next we align everything to the opinion segment labels. 

```python
>>> cmumosi_highlevel.align('Opinion Segment Labels')
```

*Opinion Segment Labels* is the key for the labels we just fetched. Since every video has multiple segments according to annotations and timing in opinion segment labels, each video will also be accompanied by a [x] where x denotes which opinion segment the computational sequence information belongs to; for example v0[2] denotes third segment of v0 (starting from [0]). 


**Word Level Alignement:**

In recent papers, it has been a common practice to perform word-level alignment. To do this with the mmdatasdk, we can do the following:

```python
>>> from mmsdk import mmdatasdk
>>> cmumosi_highlevel=mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.highlevel,'cmumosi/')
>>> cmumosi_highlevel.align('glove_vectors',collapse_functions=[myavg])
>>> cmumosi_highlevel.add_computational_sequence(mmdatasdk.cmu_mosi.labels,'cmumosi/')
>>> cmumosi_highlevel.align('Opinion Segment Labels')
```

we first aligned everything to the *glove_vectors* modality and then we align to the *Opinion Segment Labels*. Please note that with the alignment to the *glove_vectors*, we ask the align function to also collapse the other modalities. This basically means summarize the other modalities based on a set of functions. The functions all receive two argument *intervals* and *features*. Intervals is a *m times 2* and features is a *m times n* matrix. The output of the functions should be *1 times n*. For example the following function ignores intervals and just takes the average of the input features:

```python
import numpy
def myavg(intervals,features):
        return numpy.average(features,axis=0)
```

Multiple functions can be passed to *collapse_functions*, each of them will be applied one by one and will be concatenated as the final output. 

## Citations
To acquire citations for all computational sequence resources you have used simply call the bib_citations method either from an mmdataset or computational_sequence object:	

```python
>>> mydataset.bib_citations(open('mydataset.bib','w'))
>>> mycompseq.bib_citations(open('mycompseq.bib','w'))
```
	
This will output all the citations for what you have used. You may need to remove duplicates. 
