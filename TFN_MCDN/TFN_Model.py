import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.init import xavier_normal
import numpy as np

class Language_Subembedding_Network(nn.Module):
	'''Language Model'''

	def __init__(self, input_dims, hidden_dims, num_layers, dropout=0.5, bias=True):
		super(Language_Subembedding_Network, self).__init__()
		self.input_dims = input_dims
		self.hidden_dims = hidden_dims
		self.num_layers = num_layers
		self.dropout = dropout
		self.bias = bias

		output_dim = 5 #128

		self.fc_layer_1 = nn.Linear(self.hidden_dims,output_dim,bias=self.bias)	#Output of Language model is 128 dim. vector
		self.fc_layer_2 = nn.Linear(output_dim,output_dim,bias=self.bias)

		self.init_weights()

	def init_weights(self):

		self.fc_layer_1.weight = nn.init.normal(self.fc_layer_1.weight)
		self.fc_layer_1.bias = nn.init.normal(self.fc_layer_1.bias)

		self.fc_layer_2.weight = nn.init.normal(self.fc_layer_2.weight)
		self.fc_layer_2.bias = nn.init.normal(self.fc_layer_2.bias)

	def forward(self, language_input):
		
		lstm = nn.LSTM(self.input_dims,self.hidden_dims,self.num_layers)
		hidden_states = Variable(torch.randn(self.num_layers,3,self.hidden_dims))
		cell_states = Variable(torch.randn(self.num_layers,3,self.hidden_dims))

		lstm_output, hn = lstm(language_input,(hidden_states,cell_states))

		fc_output_1 = self.fc_layer_1(lstm_output)	#Fully Connected Layer 1
		fc_output_1 = F.relu(fc_output_1)	#Activation applied to Fully Connected Layer 1

		z_l = self.fc_layer_2(fc_output_1)
		z_l = F.relu(z_l)

		return z_l


class Visual_Audio_Subembedding_Network(nn.Module):	#Since both have similar fully connected configurations
	'''Visual and Audio Model'''

	def __init__(self, input_dims, bias=True):
		self.input_dims = input_dims
		self.bias = bias

		output_dim = 5

		self.fc_layer_1 = nn.Linear(self.input_dims,output_dim,bias=self.bias)
		self.fc_layer_2 = nn.Linear(output_dim,output_dim,bias=self.bias)
		self.fc_layer_3 = nn.Linear(output_dim,output_dim,bias=self.bias)

		self.init_weights()

	def init_weights(self):

		self.fc_layer_1.weight = nn.init.normal(self.fc_layer_1.weight)
		self.fc_layer_2.weight = nn.init.normal(self.fc_layer_2.weight)
		self.fc_layer_3.weight = nn.init.normal(self.fc_layer_3.weight)

		self.fc_layer_1.bias = nn.init.normal(self.fc_layer_1.bias)
		self.fc_layer_2.bias = nn.init.normal(self.fc_layer_2.bias)
		self.fc_layer_3.bias = nn.init.normal(self.fc_layer_3.bias)

	def forward(self, modality_input):

		fc_output_1 = self.fc_layer_1(Variable(modality_input))
		fc_output_1 = F.relu(fc_output_1)

		fc_output_2 = self.fc_layer_1(fc_output_1)
		fc_output_2 = F.relu(fc_output_2)

		z_modality = self.fc_layer_1(fc_output_2)
		z_modality = F.relu(fc_output_3)

		return z_modality

class Tensor_Fusion_Network(nn.Module):

	def __init__(self,language_embedding,visual_embedding,audio_embedding):

		unit_tensor = Variable(torch.ones(1))
		
		language_embedding_reshaped = language_embedding.view(-1)
		visual_embedding_reshaped = visual_embedding.view(-1)
		audio_embedding_reshaped = audio_embedding.view(-1) 

		self.z_l = torch.cat((language_embedding_reshaped, unit_tensor))
		self.z_v = torch.cat((visual_embedding_reshaped, unit_tensor))
		self.z_a = torch.cat((audio_embedding_reshaped, unit_tensor))

	def compute_cross_product(self):

		TFN = torch.ger(self.z_l,self.z_v)
		TFN = TFN.view(-1)
		TFN = torch.ger(TFN,self.z_a)
		TFN = TFN.view(-1)
		return TFN