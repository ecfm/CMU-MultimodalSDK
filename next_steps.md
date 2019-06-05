**Our next steps for the SDK**

1. Dimension names for computational sequences - done - doing final checks before commit
2. BERT, and GloVe (updates) for CMU-MOSI, CMU-MOSEI, POM - delayed due to better alignment now available
3. Releasing Social-IQ - June 15th. 
4. Intermediate goal: accomodate low-ram setups by forcing the h5py to remain on the hdd <= done, needs final testing as it broke the alignment + setitem and remove items
5. Adding passive alignment for RAVEN style models
6. Adding numpy style key search [x,1,2] => video x segment number 2, fine segment number 3


**High priority fixes**
0. A faster implementation of unify based on muliple set intersection

