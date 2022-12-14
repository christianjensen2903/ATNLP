import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn


class EncoderCell(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, rnn_type='RNN', dropout_p=0.1, device='cpu'):
        super(EncoderCell, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.RNN_type = rnn_type
        self.device = device

        self.embedding = nn.Embedding(input_size, self.hidden_size)

        self.dropout = nn.Dropout(dropout_p)

        self.rnn = nn.__dict__[self.RNN_type](
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            dropout=dropout_p
        )

    def forward(self, encoder_input, hidden):
        output = self.embedding(encoder_input).view(1, 1, -1)
        output = self.dropout(output)
        output, hidden = self.rnn(output, hidden)
        return output, hidden

    def init_hidden(self):
        if self.RNN_type == 'LSTM':
            return (
                torch.zeros(self.n_layers, 1, self.hidden_size, device=self.device),
                torch.zeros(self.n_layers, 1, self.hidden_size, device=self.device)
            )
        else:
            return torch.zeros(self.n_layers, 1, self.hidden_size, device=self.device)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, max_length=100, device='cpu', n_layers=1, rnn_type='RNN', dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.device = device
        self.all_hidden_states = torch.zeros(max_length, hidden_size, device=device)
        self.encoder_cell = EncoderCell(input_size, hidden_size, n_layers, rnn_type, dropout_p, device)

    def forward(self, input):
        encoder_hidden = self.encoder_cell.init_hidden()

        input_length = input.size(0)

        for ei in range(input_length):
            encoder_outputs, encoder_hidden = self.encoder_cell(input[ei], encoder_hidden)
            self.all_hidden_states[ei] = encoder_hidden[0][0].detach()

        return encoder_outputs, encoder_hidden


class DecoderCell(nn.Module):
    def __init__(self, output_size, hidden_size, n_layers=1, rnn_type='RNN', dropout_p=0.1, device='cpu', max_length=100):
        super(DecoderCell, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.output_size = output_size
        self.RNN_type = rnn_type

        self.embedding = nn.Embedding(output_size, self.hidden_size)

        self.dropout = nn.Dropout(dropout_p)

        self.rnn = nn.__dict__[self.RNN_type](
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            dropout=dropout_p
        )

        self.out = nn.Linear(self.hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, decoder_input, hidden):
        output = self.embedding(decoder_input).view(1, 1, -1)
        output = self.dropout(output)
        output = F.relu(output)

        output, hidden = self.rnn(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

class AdditiveAttention(nn.Module):
    """Additive attention."""

    def __init__(self, key_size, query_size, num_hiddens, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)

    def forward(self, queries, keys, values):
        queries, keys = self.W_q(queries), self.W_k(keys)
        features = queries + keys
        features = torch.tanh(features)
        scores = self.w_v(features)
        self.attention_weights = F.softmax(scores, dim=0).reshape((-1, 1, 1))
        
        bmm = torch.bmm(self.attention_weights, values)
        return torch.sum(bmm, dim=0)

class AttnDecoderCell(nn.Module):
    def __init__(self, ouput_size, hidden_size, num_layers, rnn_type,
                 dropout_p=0, device='cpu', max_length=100):
        super().__init__()
        self.attention = AdditiveAttention(num_hiddens=hidden_size, key_size=hidden_size, query_size=hidden_size)
        self.embedding = nn.Embedding(ouput_size, hidden_size)
        
        self.rnn = nn.__dict__[rnn_type](
            input_size=2*hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_p
        )
        
        self.dense = nn.Linear(hidden_size, ouput_size)
        self.dropout = nn.Dropout(dropout_p)
        #self.attention_weights = []

    def forward(self, input, hidden_state, enc_outputs):
        
        enc_outputs = enc_outputs.unsqueeze(0).permute(1, 0, 2)

        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        embedded = F.relu(embedded)

        query = torch.unsqueeze(hidden_state[-1], dim=1)
        context = self.attention(
            query, enc_outputs, enc_outputs)

        context = context.unsqueeze(0)

        x = torch.cat((context, embedded), dim=-1)

        outputs, hidden_state = self.rnn(x, hidden_state)
        #self._attention_weights.append(self.attention.attention_weights)

        outputs = F.log_softmax(self.dense(outputs[0]), dim=1)
        return outputs, hidden_state

    @property
    def attention_weights(self):
        return self._attention_weights


class DecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, n_layers=1, rnn_type='RNN', dropout_p=0.1, attention=False,
                 device='cpu', max_length=100):
        super(DecoderRNN, self).__init__()


        self.attention = attention
        if attention:
            self.decoder_cell = AttnDecoderCell(output_size, hidden_size, n_layers, rnn_type, dropout_p, device, max_length)
        else:
            self.decoder_cell = DecoderCell(output_size, hidden_size, n_layers, rnn_type, dropout_p, device)

    def forward(self, input, hidden, enc_outputs=None):
        assert enc_outputs is not None if self.attention else True  # If attention is used, all encoder hidden states must be provided
        if self.attention:
            output, hidden = self.decoder_cell(input, hidden, enc_outputs)
        else:
            output, hidden = self.decoder_cell(input, hidden)

        return output, hidden
