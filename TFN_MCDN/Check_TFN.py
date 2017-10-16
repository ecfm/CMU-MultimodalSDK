from TFN_Model import Language_Subembedding_Network, Visual_Audio_Subembedding_Network, Tensor_Fusion_Network
import torch
from torch import nn
from torch.autograd import Variable

lang_input = Variable(torch.randn(5,3,10))
lang_model = Language_Subembedding_Network(10,20,2)
z_l = lang_model.forward(lang_input)

visual_input = Variable(torch.randn(5,3,10))
visual_model = Visual_Audio_Subembedding_Network(10)
z_v = visual_model.forward(visual_input)

audio_input = Variable(torch.randn(5,3,10))
audio_model = Visual_Audio_Subembedding_Network(10)
z_a = audio_model.forward(audio_input)

