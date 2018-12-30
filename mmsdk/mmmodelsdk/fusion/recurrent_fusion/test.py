import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from model import RecurrentFusion
import numpy


#in=40, 3 modalities,4 attentions
full_in=numpy.array(numpy.zeros([32,40]))
inputx=Variable(torch.Tensor(full_in),requires_grad=True)
full_in=numpy.array(numpy.zeros([32,12]))
inputy=Variable(torch.Tensor(full_in),requires_grad=True)
full_in=numpy.array(numpy.zeros([32,20]))
inputz=Variable(torch.Tensor(full_in),requires_grad=True)
modalities=[inputx,inputy,inputz]

fmodel=RecurrentFusion([40,12,20],100)

out=fmodel.fusion(modalities,steps=5)
print(out[0].shape,out[1].shape,out[2].shape,sum(out[0][4,:,:]-out[1]))

#a=numpy.array([[1,2,3],[4,5,6]])
#b=Variable(torch.Tensor(a))
#c=b.repeat(1,2)
#print(c.shape)
#print(F.softmax(c,dim=1))
#print(c.view(2,2,3)[0,1,:])


