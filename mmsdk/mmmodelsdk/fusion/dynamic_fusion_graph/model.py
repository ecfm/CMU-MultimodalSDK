#CMU Multimodal SDK, CMU Multimodal Model SDK

#Tensor Fusion Network for Multimodal Sentiment Analysis, Amir Zadeh, Minghai Chen, Soujanya Poria, Erik Cambria, Louis-Philippe Morency - https://arxiv.org/pdf/1707.07250.pdf

#in_modalities: is a list of inputs from each modality - the first dimension of all the modality inputs must be the same, it will be the batch size. The second dimension is the feature dimension. There are a total of n modalities.

#out_dimension: the output of the tensor fusion

import torch
import time
from torch import nn
import torch.nn.functional as F
import copy
from six.moves import reduce
from itertools import chain,combinations
from collections import OrderedDict

class DynamicFusionGraph(nn.Module):

        def __init__(self,efficacy_model,pattern_model,in_dimensions,out_dimension):
                super(DynamicFusionGraph, self).__init__()

		self.num_modalities=len(in_dimensions)
		#in this part we sort out number of connections, how they will be connected etc.
		powerset=list(chain.from_iterable(combinations(range(self.num_modalities), r) for r in range(self.num_modalities+1)))[1:]

		#initializing the models inside the DFG
		input_shapes=OrderedDict([(key,value) for key,value in zip(range(self.num_modalities),in_dimensions)])
		networks={}
		outputs={}
		for key in powerset[self.num_modalities:]:
			#connections coming from the unimodal components
			unimodal_dims=0
			for modality in key:
				in_dim+=in_dimensions[modality]
			multimodal_dims=((2**len(key)-2)-len(key))*pattern_out_dimensions
			#for the network that outputs key component, what is the input dimension
			final_dims=unimodal_dims+multimodal_dims
			input_shapes[key]=final_dims
			pattern_copy=copy.deepcopy(pattern_model)
			final_model=nn.Sequential([nn.Linear(input_shapes[key],list(pattern_copy.children())[0].in_featuers),pattern_copy)
			networks[key]=final_model
		#finished construction weights, now onto the t_network which summarizes the graph
		t_in_dimension=sum(in_dimensions)
		
		
		t_network=

	def fusion(self,in_modalities):

		bs=in_modalities[0].shape[0]
		tensor_product=in_modalities[0]
		
		#calculating the tensor product
		
		for in_modality in in_modalities[1:]:
			tensor_product=torch.bmm(tensor_product.unsqueeze(2),in_modality.unsqueeze(1))
			tensor_product=tensor_product.view(bs,-1)
		
		return self.linear_layer(tensor_product)

        def forward(self, x):
		print("Not yet implemented for nn.Sequential")
		exit(-1)

if __name__=="__main__":
	print("This is a module and hence cannot be called directly ...")
	exit(-1)

