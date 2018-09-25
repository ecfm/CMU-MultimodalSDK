from mmsdk.mmdatasdk import log, computational_sequence
import sys
import numpy
import time
from tqdm import tqdm
import os

epsilon=10e-4

class mmdataset:

	def __init__(self,recipe,destination=None):
		
		self.computational_sequences={}	

		if type(recipe) is not dict:
			log.error("Dataset recipe must be a dictionary type object ...")
		
		for entry, address in recipe.items():
			self.computational_sequences[entry]=computational_sequence(address,destination)
	
	def add_computational_sequences(self,recipe,destination):
		for entry, address in recipe.iteritems():
			if entry in self.computational_sequences:
				log.error("Dataset already contains <%s> computational sequence"%entry)
			self.computational_sequences[entry]=computational_sequence(address,destination)

	def bib_citations(self,outfile=None):
		
		outfile=sys.stdout if outfile is None else outfile
		sdkbib='@article{zadeh2018multi, title={Multi-attention recurrent network for human communication comprehension}, author={Zadeh, Amir and Liang, Paul Pu and Poria, Soujanya and Vij, Prateek and Cambria, Erik and Morency, Louis-Philippe}, journal={arXiv preprint arXiv:1802.00923}, year={2018}}'
		outfile.write('mmsdk bib: '+sdkbib+'\n\n')
		for entry,compseq in self.computational_sequences.items():
			compseq.bib_citations(outfile)

	def align(self,reference,collapse_functions=None,replace=True):
		aligned_output={}
		for sequence_name in self.computational_sequences.keys():
			aligned_output[sequence_name]={}
		if reference not in self.computational_sequences.keys():
			log.error("Computational sequence <%s> does not exist in dataset"%reference,error=True)
		refseq=self.computational_sequences[reference].data
		#this for loop is for entry_key - for example video id or the identifier of the data entries
		log.status("Alignment based on <%s> computational sequence started ..."%reference)
		pbar = tqdm(total=len(refseq.keys()),unit=" Computational Sequence Entries",leave=False)
		pbar.set_description("Overall Progress")
		for entry_key in list(refseq.keys()):
			pbar_small=tqdm(total=refseq[entry_key]['intervals'].shape[0],unit=" Segments",leave=False)
			pbar_small.set_description("Aligning %s"%entry_key)
			for i in range(refseq[entry_key]['intervals'].shape[0]):	
				#interval for the reference sequence
				ref_time=refseq[entry_key]['intervals'][i,:]
				#we drop zero or very small sequence lengths - no align for those
				if (abs(ref_time[0]-ref_time[1])<epsilon):
					pbar_small.update(1)
					continue

				#aligning all sequences (including ref sequence) to ref sequence
				for otherseq_key in list(self.computational_sequences.keys()):

					intersects,intersects_features=self.__intersect_and_copy(entry_key,ref_time,self.computational_sequences[otherseq_key],epsilon)
					#there were no intersections between reference and subject computational sequences for the entry
					if intersects.shape[0] == 0:
						continue
					#collapsing according to the provided functions	
					if type(collapse_functions) is list:
						intersects,intersects_features=self.__collapse(intersects,intersects_features,collapse_functions)
					if(intersects.shape[0]!=intersects_features.shape[0]):
						log.error("Dimension mismatch between intervals and features when aligning <%s> computational sequences to <%s> computational sequence"%(otherseq_key,reference))
					aligned_output[otherseq_key][entry_key+"[%d]"%i]={}
					aligned_output[otherseq_key][entry_key+"[%d]"%i]["intervals"]=intersects
					aligned_output[otherseq_key][entry_key+"[%d]"%i]["features"]=intersects_features
				pbar_small.update(1)
			pbar_small.close()
			pbar.update(1)
		pbar.close()
		log.success("Alignment to <%s> complete."%reference)
		if replace is True:
			log.status("Replacing dataset content with aligned computational sequences")
			self.__set_computational_sequences(aligned_output)
			return None
		else:
			log.status("Creating new dataset with aligned computational sequences")
			newdataset=mmdataset({})
			newdataset.__set_computational_sequences(aligned_output)
			return newdataset	
	
	def __collapse(self,intervals,features,functions):
		#we simply collapse the intervals to (1,2) matrix
		new_interval=numpy.array([[intervals.min(),intervals.max()]])
		try:
			new_features=numpy.concatenate([function(intervals,features) for function in functions],axis=0)
			if len(new_features.shape)==1:
				new_features=new_features[None,:]
		except:
			log.error("Cannot collapse given the set of function.", error=True)
		return new_interval,new_features
			
	def __set_computational_sequences(self,new_computational_sequences_data):
		self.computational_sequences={}
		for sequence_name in list(new_computational_sequences_data.keys()):
			self.computational_sequences[sequence_name]=computational_sequence(sequence_name)
			self.computational_sequences[sequence_name].setData(new_computational_sequences_data[sequence_name],sequence_name)
			self.computational_sequences[sequence_name].rootName=sequence_name
				
	def deploy(self,destination,filenames):
		if os.path.isdir(destination) is False:
			os.mkdir(destination)
		for seq_key in list(self.computational_sequences.keys()):
			if seq_key not in list(filenames.keys()):
				log.error("Filename for %s computational sequences not specified"%seq_key)
			filename=filenames[seq_key]
			if filename [:-4] != '.csd':
				filename+='.csd'
			self.computational_sequences[seq_key].deploy(os.path.join(destination,filename))
		
	def __intersect_and_copy(self,ref_entry_key,ref,sub_compseq,epsilon):
		relevant_entries=[x for x in sub_compseq.data.keys() if x.split('[')[0]==ref_entry_key]
		sub=numpy.concatenate([sub_compseq.data[x]["intervals"] for x in relevant_entries],axis=0)
		features=numpy.concatenate([sub_compseq.data[x]["features"] for x in relevant_entries],axis=0)
	        #copying and inverting the ref
	        ref_copy=ref.copy()
	        ref_copy[1]=-ref_copy[1]
	        ref_copy=ref_copy[::-1]
       		sub_copy=sub.copy()
       		sub_copy[:,0]=-sub_copy[:,0]
		#finding where intersect happens
	        where_intersect=(numpy.all((sub_copy-ref_copy)>(-epsilon),axis=1)==True)
	        intersectors=sub[where_intersect,:]
		intersectors=numpy.concatenate([numpy.maximum(intersectors[:,0],ref[0])[:,None],numpy.minimum(intersectors[:,1],ref[1])[:,None]],axis=1)
		intersectors_features=features[where_intersect,:]
		#checking for boundary cases and also zero length
		where_nonzero_len=numpy.where(abs(intersectors[:,0]-intersectors[:,1])>epsilon)
		intersectors_final=intersectors[where_nonzero_len]
		intersectors_features_final=intersectors_features[where_nonzero_len]
	        return intersectors_final,intersectors_features_final

	def unify():
		pass

	
