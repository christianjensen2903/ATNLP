# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from io import open
# import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device( "cpu")
SOS_token = 0
EOS_token = 1

USE_CUDA=False
# Further, they actually used the "concat" strategy. So this should be self.attn = Attn("concat", hidden_size)
def printShape(t):
    print(t.shape)

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attnLinear = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat': # use this
            self.attnLinear = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        printShape(hidden)
        printShape(encoder_outputs)
        # torch.Size([1, 100])
        # torch.Size([3, 100])

        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies_e_ij = (torch.zeros(this_batch_size, max_len)) # B x S

        if USE_CUDA:
            attn_energies_e_ij = attn_energies_e_ij.cuda()

        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies_e_ij[b, i] = self.e_ij(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        # Normalize e_ij to get a_ij, to weights in range 0 to 1, resize to 1 x B x S
        a_ij=F.softmax(attn_energies_e_ij).unsqueeze(1)
        return a_ij

    def e_ij(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            energy = self.attnLinear(encoder_output)
            energy = hidden.dot(energy)
            return energy

        elif self.method == 'concat':
            print(self.v.shape)
            printShape(hidden)
            printShape(encoder_output)
            print(self.attnLinear(torch.cat((hidden, encoder_output), 1)).shape)
            energy = self.v.dot(self.attnLinear(torch.cat((hidden, encoder_output), 1))) # v * tanH(W*s+U*hj)
            return energy


# https://github.com/spro/practical-pytorch/blob/c520c52e68e945d88fff563dba1c028b6ec0197b/seq2seq-translation/seq2seq-translation-batched.ipynb
class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length,n_layers=1, dropout_p=0.1):
        super(BahdanauAttnDecoderRNN, self).__init__()

        # Define parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attn_a_ij = Attn("concat", hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, word_input, last_hidden, encoder_outputs):
        # forward for a single decoder time step, but will use all encoder outputs

        # Get the embedding of the current input word (last output word)
        word_embedded = self.dropout(self.embedding(word_input).view(1, 1, -1)) # S=1 x B x N

        # Calculate attention weights and apply to encoder outputs
        attn_weights_a_ij = self.attn_a_ij(last_hidden[-1], encoder_outputs)
        context_c_i = attn_weights_a_ij.bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N

        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((word_embedded, context_c_i), 2)
        output, hidden = self.gru(rnn_input, last_hidden)

        # Final output layer
        output = output.squeeze(0) # B x N
        output = F.log_softmax(self.out(torch.cat((output, context_c_i), 1)))

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights_a_ij


