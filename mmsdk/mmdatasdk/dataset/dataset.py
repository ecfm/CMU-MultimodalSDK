from mmsdk.mmdatasdk import log, computational_sequence
import sys

class mmdataset:
	def __init__(self,recipe,destination=None):
		
		self.computational_sequences={}	

		if type(recipe) is not dict:
			log.error("Dataset recipe must be a dictionary type object ...")
		
		for entry, address in recipe.iteritems():
			print(entry,address,destination)
			self.computational_sequences[entry]=computational_sequence(address,destination)
		
	def bib_citations(self,outfile=None):
		
		outfile=sys.stdout if outfile is None else outfile
		sdkbib='@article{zadeh2018multi, title={Multi-attention recurrent network for human communication comprehension}, author={Zadeh, Amir and Liang, Paul Pu and Poria, Soujanya and Vij, Prateek and Cambria, Erik and Morency, Louis-Philippe}, journal={arXiv preprint arXiv:1802.00923}, year={2018}}'
		outfile.write('mmsdk bib: '+sdkbib+'\n\n')
		for entry,compseq in self.computational_sequences.iteritems():
			compseq.bib_citations(outfile)
		
	
