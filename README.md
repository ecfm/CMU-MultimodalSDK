# News

CMU-Multimodal SDK V 1.0.0 is released. Please be advised of major changes to the data structures due to improvements to data loading and downloading. 

# CMU-Multimodal SDK Version 1.0.0 (mmsdk)

CMU-Multimodal SDK provides tools to easily load well-known multimodal datasets and rapidly build neural multimodal deep models. Hence the SDK comprises of two modules: 1) mmdatasdk: module for downloading and procesing multimodal datasets using computational sequences. 2) mmmodelsdk: tools to utilize complex neural models as well as layers for building new models. 

## 1. CMU Multimodal Data SDK (mmdatasdk)

CMU-Multimodal Data SDK simplifies downloading nad loading multimodal datasets. The module mmdatasdk treats each multimodal dataset as a combination of **computational sequences**. Each computational sequence contains information from one modality in a heirarchical format, defined in the continuation of this section. Computational sequences are self-contained and independent; they can be used to train models in isolation. They can be downloaded, shared and registered with our trust servers. This allows the community to share data and recreate results in a more elegant way using computational sequence intrgrity checks. Furthermore, this integrity check allows users to download the correct computational sequences. 

Each computational sequence is a heirarchical data strcuture which contains two key elements 1) "data" is a heirarchy of features in the computational sequence categorized based on unique multimodal source identifier (for example video id). Each multimodal source has two matrices associated with it: features and intervals. Features denote the computational descriptors and intervals denote their associated timestamp. Both features and intervals are numpy 2d arrays. 2) "metadata": contains information about the computational sequence including integrity and version information. The computational sequences are stored as hdf5 onjects on hard disk with ".csd" extension (computational sequential data). Both the data and metadata are stored under "root name" (root of the heirarchy)

A dataset is defined as a dictionary of multiple computational sequences. Entire datasets can be shared using recipes as opposed to old-fashioned dropbox links or ftp servers. Computational sequences are downloaded one by one and their individual integrity is checked to make sure they are the ones users wanted to share. Users can register their extracted features with our trust server to use this feature. They can also request storage of their features on our servers 




## 2. Installation

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
pip install h5py validators tqdm numpy 
```

## 3. Usage

The first step in most machine learning tasks is to acquire the data. 

```python
>>> from mmsdk import mmdatasdk
```

Now that mmdatasdk is loaded you can proceed to fetch a dataset. Let's assume the server that hosts the dataset is host.edu and trust server is trust.edu (list of server links for each dataset provided later in the readme). 
If you are using a standard featureset from provided datasets you can simply download them. 

```python
>>> from mmdatasdk import computational_sequence
>>> mycompseq=computational_sequence('host.edu/arbitrarycompseq.csd','mypath/name.csd')
```

This will download an arbitrary computational sequence (arbitrarycompseq.csd) and stores it as name.csd in mypath. If downloading a dataset from a link, mmdatasdk always checks integrity with trust server. Another usage of computational_sequence is if you have a pre-exising file in storage:

```python
>>> from mmdatasdk import computational_sequence
>>> mycompseq=computational_sequence('mypath/name.csd')
```

This will simply load the file name.csd from mypath. This will also force a trust check with the trust server. 

You can also initialize an empty computational_sequence using the following: 

```python
>>> from mmdatasdk import computational_sequence
>>> mycompseq=computational_sequence('myrootname')
```

This comes in handy if you are building a computational sequence from scratch (more advanced stuff, discussed later). You cannot register a computational sequence with our trust server unless the computational sequence passes both data and metadata integrity checks (to make sure both are in the correct format). 

In most cases you won't need to deal with computational_sequence but rather with mmdataset. The scripts below are examples of downloading datasets to the mycmu_*_dir. 

```python
>>> from mmdatasdk import mmdataset
>>> cmumosei_highlevel=mmdataset(mmdatasdk.cmu_mosei.highlevel,'mycmu_mosei_dir')
>>> cmumosi_highlevel=mmdataset(mmdatasdk.cmu_mosi.highlevel,'mycmu_mosi_dir')
```

This script will download high-level CMU-MOSEI features according to highlevel receipe. Each recipe is a key-value dictionary with key as the name you would like to refer to the computational sequence as (different than root name) and value is the link to download the computational seqeuence from. You can find the standard datasets in the /dataset/standard_datasets/ folder. 

The computational sequences inside a mmdataset can be aligned with each other according to a heirarchy. A heirarchy is an instance of computational sequence that does not have features inside its data, but just intervals. 


## Citations
To acquire citations for all computational sequence resources you have used simply call the bib_citations method either from an mmdataset or computational_sequence object:	

```python
>>> mydataset.bib_citations(open('mydataset.bib','w'))
>>> mycompseq.bib_citations(open('mycompseq.bib','w'))
```
	
This will output all the citations for what you have used. 
