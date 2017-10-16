import torch, csv
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from MCDN_TFN_Model_With_Different_LSTM_Dim import MCDN
from TFN_Model import Tensor_Fusion_Network

#Add config file parameters for each param mentioned below
input_dim = 2
time = 3
modality = 3
input_seq = []
target = Variable(torch.FloatTensor([3.5]))
epochs = 10
learning_rate = 0.1
nattentions = 6
optimizer = ["SGD","Adam"]

for e in range(epochs):

	for t in range(time):
	    step = []
	    for m in range(modality):
	        step.append(Variable(torch.randn(1, input_dim)))	#Input_Data from MOSI
	    input_seq.append(step)

	mcdn = MCDN([2,2,2], [20,20,20], 15, nattentions)

	final_layer = nn.Linear(15, 1)

	loss = nn.MSELoss()

	optimizer = optim.SGD(mcdn.parameters(), lr=learning_rate)
	
	mcdn.zero_grad()

	(hidden, zx) = mcdn(input_seq)
	output = final_layer(zx)

	loss_val = loss(output, target)
	print loss_val

	loss_val.backward()
	optimizer.step()

#Hyperparameters being used: SGD learning rate, Time Steps, Epochs, no. of attentions
# writer.writerow([])