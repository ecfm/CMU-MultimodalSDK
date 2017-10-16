import torch.nn as nn
import torch
from torch.autograd import Variable


class LSTHMCell(nn.Module):
    """Long-short term hybrid memory cell"""

    def __init__(self, input_dim, hidden_dim, fusion_dim, dropout=0.5, bias=True):
        '''
        Initialize a LSTHM cell.
        :param hidden_dim: the dimensions of hidden layers of LSTM
        :param input_dim: the dimensions of the input
        :param dropout: dropout rate
        :param fusion_dim: dimension of the externally provided fusion vector zx
        '''
        super(LSTHMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.dropout = dropout

        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(input_dim + hidden_dim + fusion_dim, 4 * hidden_dim)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.init_weights()


    def init_weights(self):
        """Initialize the weights of each modality LSTMs, TO BE CHANGED TO XAVIER INIT"""
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, input, hx, cx, zx):
        """
        Computes the next hx using input at current step and hx, zx from last step
        :param input: input at this time step
        :param hx: hidden state from last time step
        :param zx: fusion state from MAB
        :return: new hidden state and cell state
        """
        gates = self.linear(torch.cat([input, hx, zx], dim=1))
        (i, f, o, g) = torch.split(gates, self.hidden_dim, dim=1) # split the gates
        i = self.sigmoid(i)
        f = self.sigmoid(f)
        o = self.sigmoid(o)
        g = self.tanh(g)
        c_cur = f * cx + i * g
        h_cur = o * self.tanh(c_cur)
        return (h_cur, c_cur)


class MCDN(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, input_dim, lstm_dim, fusion_dim, nattentions, dropout_lstm=0.5):
        '''
        Initialize a LSTHM cell.
        :param lstm_dim: the dimensions of hidden layers of LSTM
        :param input_dim: the dimensions of the inputs of different modalities, as a list [dim1, dim2, dim3]
        :param dropout: dropout rate
        :param nattentions: number of attentions to use
        '''
        super(MCDN, self).__init__()
        self.input_dim = input_dim
        self.lstm_dim = lstm_dim
        self.dropout_lstm = dropout_lstm
        self.fusion_dim = fusion_dim
        self.nattentions = nattentions
        self.modality = len(input_dim) # number of modalities
        self.LSTHMCells = [] # list of LSTM cells for each modality
        self.LSTHMDropouts = [] # list of dropout layers for each modality LSTM
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        # Define the LSTMs for all modalities
        for i in range(self.modality):
            self.LSTHMCells.append(LSTHMCell(self.input_dim[i], self.lstm_dim, self.fusion_dim))
            self.LSTHMDropouts.append(nn.Dropout(self.dropout_lstm))

        # Define the multi-attentions
        self.attentions = []
        for i in range(nattentions):
            self.attentions.append(nn.Linear(self.lstm_dim * self.modality, self.lstm_dim * self.modality))

        # Define the final weights of the network computes Z_t
        self.Ws = nn.Linear(self.lstm_dim * self.modality * self.nattentions, self.modality * self.lstm_dim)
        self.Wz = nn.Linear(self.lstm_dim * self.modality, self.fusion_dim)
        self.init_weights()


    def init_weights(self):
        """Weight Initialization: TO BE MODIFIED TO USE XAVIER INITIALIZATION"""
        pass


    def forward(self, input_seq):
        """
        Forward pass
        :param input_seq: a list of list of Variables, the outer list is the time steps, inner is the list for modalities
        :return: final hidden
        """
        # Initialize hidden states for all LSTMs
        hidden = []
        cell_state = []
        zx = Variable(torch.zeros(1, self.fusion_dim)) # zx is shared for all modalities

        for i in range(self.modality):
            hidden.append(Variable(torch.zeros(1, self.lstm_dim)))
            cell_state.append(Variable(torch.zeros(1, self.lstm_dim)))

        # Computation
        for time_step in input_seq:

            # Compute the hidden states and concat them
            for i, modality_input in enumerate(time_step):
                hidden[i], cell_state[i] = self.LSTHMCells[i](modality_input, hidden[i], cell_state[i], zx)
            hidden_all = torch.cat(hidden, dim=1)


            # Compute the attention weights and attended output and concat them
            attended_hidden = []
            for attention in self.attentions:
                attended_hidden.append(hidden_all * self.softmax(attention(hidden_all)))
            attended_all = torch.cat(attended_hidden, dim=0)

            # Split and reshape the concatenated attended hidden states
            reshaped_attended = []
            resplitted_attended = torch.split(attended_all, self.lstm_dim, dim=1)
            for attended in resplitted_attended:
                reshaped_attended.append(attended.contiguous().view(1, -1)) # Stretch each one flat
            reshaped_all = torch.cat(reshaped_attended, dim=1)

            # Compute the final Z_t
            sx = self.Ws(reshaped_all)
            zx = self.Wz(sx)

        return hidden, zx