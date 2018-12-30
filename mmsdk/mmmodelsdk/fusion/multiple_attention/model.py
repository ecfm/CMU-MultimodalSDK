#CMU Multimodal SDK, CMU Multimodal Model SDK

#Multi-attention Recurrent Network for Human Communication Comprehension, Amir Zadeh, Paul Pu Liang, Soujanya Poria, Erik Cambria, Prateek Vij, Louis-Philippe Morency - https://arxiv.org/pdf/1802.00923.pdf

#in_modalities: is a list of inputs from each modality - the first dimension of all the modality inputs must be the same, it will be the batch size. The second dimension is the feature dimension. There are a total of n modalities.

#attention_model: is a pytorch nn.Sequential which takes in an input with size (bs * m0+...+mn) with m_i being the dimensionality of the features in modality i. Output is the (bs * (m0+...+mn)*num_atts).

#dim_reduce_nets: is a list of pytorch nn.Sequential which takes in an input with size (bs*(mi*num_atts))
#num_atts is the number of attentions

#num_atts: number of attentions


import torch
import time
from torch import nn
import torch.nn.functional as F


class MultipleAttentionFusion(nn.Module):

        def __init__(self,attention_model,dim_reduce_nets,num_atts):
                super(MultipleAttentionFusion, self).__init__()
		self.attention_model=attention_model
		self.dim_reduce_nets=dim_reduce_nets
		self.num_atts=num_atts

	def fusion(self,in_modalities):

		#getting some simple integers out
		num_modalities=len(in_modalities)
		#simply the tensor that goes into attention_model
		in_tensor=torch.cat(in_modalities,dim=1)
		#calculating attentions
		atts=F.softmax(self.attention_model(in_tensor),dim=1)
		#calculating the tensor that will be multiplied with the attention
		out_tensor=torch.cat([in_modalities[i].repeat(1,self.num_atts) for i in range(num_modalities)],dim=1)
		#calculating the attention
		att_out=atts*out_tensor
	
		#now to apply the dim_reduce networks
		#first back to however modalities were in the problem
		start=0
		out_modalities=[]
		for i in range(num_modalities):
			modality_length=in_modalities[i].shape[1]*self.num_atts
			out_modalities.append(att_out[:,start:start+modality_length])
			start=start+modality_length
	
		#apply the dim_reduce
		dim_reduced=[self.dim_reduce_nets[i](out_modalities[i]) for i in range(num_modalities)]
		#multiple attention done :)
		return dim_reduced

        def forward(self, x):
		print("Not yet implemented for nn.Sequential")
		exit(-1)



if __name__=="__main__":
	print("This is a module and hence cannot be called directly ...")
	exit(-1)

