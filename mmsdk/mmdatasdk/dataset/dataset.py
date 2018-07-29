from mmsdk.mmdatasdk import log, computational_sequence
import sys
import numpy
import time
from tqdm import tqdm

epsilon=10e-5

class mmdataset:
	def __init__(self,recipe,destination=None):
		
		self.computational_sequences={}	

		if type(recipe) is not dict:
			log.error("Dataset recipe must be a dictionary type object ...")
		
		for entry, address in recipe.items():
			self.computational_sequences[entry]=computational_sequence(address,destination)
		
	def bib_citations(self,outfile=None):
		
		outfile=sys.stdout if outfile is None else outfile
		sdkbib='@article{zadeh2018multi, title={Multi-attention recurrent network for human communication comprehension}, author={Zadeh, Amir and Liang, Paul Pu and Poria, Soujanya and Vij, Prateek and Cambria, Erik and Morency, Louis-Philippe}, journal={arXiv preprint arXiv:1802.00923}, year={2018}}'
		outfile.write('mmsdk bib: '+sdkbib+'\n\n')
		for entry,compseq in self.computational_sequences.items():
			compseq.bib_citations(outfile)

	#TODO: this is implentation #1. update with new implentation later 
	def align(self,reference,replace=True):
		aligned_output={}
		for sequence_name in self.computational_sequences.keys():
			aligned_output[sequence_name]={}
		if reference not in self.computational_sequences.keys():
			log.error("Computational sequence %s does not exist in dataset"%reference,error=True)
		refseq=self.computational_sequences[reference].data
		#this for loop is for entry_key - for example video id or the identifier of the data entries
		log.status("Alignment based on %s computational sequence started ..."%reference)
		pbar = tqdm(total=len(refseq.keys()),unit=" Computational Sequence Entries")
		pbar.set_description("Overall Progress")
		for entry_key in list(refseq.keys()):
			pbar_small=tqdm(total=refseq[entry_key]['intervals'].shape[0])
			pbar_small.set_description("Aligning %s"%entry_key)
			#intervals for the reference sequence
			for i in range(refseq[entry_key]['intervals'].shape[0]):
				ref_time=refseq[entry_key]['intervals'][i,:]
				if (abs(ref_time[0]-ref_time[1])<epsilon):
					pbar_small.update(1)
					continue
				#aligning all sequences (including ref sequence) to ref sequence
				for otherseq_key in list(self.computational_sequences.keys()):
					otherseq=self.computational_sequences[otherseq_key].data[entry_key]
					#list to contain intersection for (otherseq_key,i)
					list_intervals=[]
					list_features=[]
					#checking all intervals of the otherseq for intersection
					for j in range(otherseq["intervals"].shape[0]):
						sub_time=otherseq["intervals"][j]
						this_features=otherseq["features"][j,:]
						intersect,intersect_start,intersect_end=self.__intersect(ref_time,sub_time)
						if intersect == True:
							list_intervals.append([intersect_start,intersect_end])
							list_features.append(this_features)
					
					aligned_output[otherseq_key][entry_key+"[%d]"%i]={}
					aligned_output[otherseq_key][entry_key+"[%d]"%i]["intervals"]=numpy.array(list_intervals,dtype='float32')
					aligned_output[otherseq_key][entry_key+"[%d]"%i]["features"]=numpy.array(list_features,dtype='float32')
					if (len(aligned_output[otherseq_key][entry_key+"[%d]"%i]["intervals"].shape)!=2):
						print ("Fuck")
						print (aligned_output[otherseq_key][entry_key+"[%d]"%i]["intervals"].shape)
						print (aligned_output[otherseq_key][entry_key+"[%d]"%i]["features"].shape)
						print (ref_time,i)
						print (refseq[entry_key]['features'][i,:].shape)
						time.sleep(10)
				pbar_small.update(1)
			pbar_small.visible=False
			pbar_small.close()
			pbar.update(1)
		pbar.visible=False
		pbar.close()
		log.success("Alignment to %s done."%reference)
		if replace is True:
			log.status("Replacing dataset content with aligned computational sequences")
			self.__set_computational_sequences(aligned_output)
			return None
		else:
			log.status("Creating new dataset with aligned computational sequences")
			newdataset=mmdataset({})
			newdataset.__set_computational_sequences(aligned_output)
			return newdataset	
		print()
	
	def __set_computational_sequences(self,new_computational_sequences_data):
		self.computational_sequences={}
		for sequence_name in list(new_computational_sequences_data.keys()):
			self.computational_sequences[sequence_name]=computational_sequence(sequence_name)
			self.computational_sequences[sequence_name].setData(new_computational_sequences_data[sequence_name],sequence_name)
				

	def __intersect(self,ref_time,sub_time):
		s0,e0=ref_time
		s1,e1=sub_time
		#intersection
		if e1-s0>=epsilon and e0-s1>=epsilon:
			intersect_start=max(s0,s1)
			intersect_end=min(e0,e1)
			#just boundary case
			if intersect_start==intersect_end:
				return False,None,None
			return True,intersect_start,intersect_end
		#no intersection
		else:
			return False,None,None
	def unify():
		pass

	
