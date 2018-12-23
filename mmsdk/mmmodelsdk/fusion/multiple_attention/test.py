import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from model import MultipleAttentionFusion
import numpy


#in=40, 3 modalities,4 attentions
full_in=numpy.array(numpy.zeros([32,40]))
inputx=Variable(torch.Tensor(full_in),requires_grad=True)
full_in=numpy.array(numpy.zeros([32,12]))
inputy=Variable(torch.Tensor(full_in),requires_grad=True)
full_in=numpy.array(numpy.zeros([32,20]))
inputz=Variable(torch.Tensor(full_in),requires_grad=True)
modalities=[inputx,inputy,inputz]

my_attention =	nn.Sequential(nn.Linear(72,72*4))
small_netx =	nn.Sequential(nn.Linear(160,10))
small_nety =	nn.Sequential(nn.Linear(48,20))
small_netz =	nn.Sequential(nn.Linear(80,30))
smalls_nets=[small_netx,small_nety,small_netz]
fmodel=MultipleAttentionFusion(my_attention,smalls_nets,4)

out=fmodel.fusion(modalities)
print([o.shape for o in out])

#a=numpy.array([[1,2,3],[4,5,6]])
#b=Variable(torch.Tensor(a))
#c=b.repeat(1,2)
#print(c.shape)
#print(F.softmax(c,dim=1))
#print(c.view(2,2,3)[0,1,:])


