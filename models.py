import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, input_size, hidden_size, device='cpu', n_layers=1, rnn_type='RNN', dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.device = device
        self.encoder_cell = EncoderCell(input_size, hidden_size, n_layers, rnn_type, dropout_p, device)

    def forward(self, input):
        encoder_hidden = self.encoder_cell.init_hidden()

        input_length = input.size(0)

        encoder_hidden_all = torch.zeros(input_length, self.encoder_cell.hidden_size,
                                         device=self.device)  # Stores all hidden states

        for ei in range(input_length):
            _, encoder_hidden =  self.encoder_cell(
                input[ei], encoder_hidden)
            if  self.encoder_cell.RNN_type == 'LSTM':
                encoder_hidden_all[ei] = encoder_hidden[0][-1, 0]
            else:
                encoder_hidden_all[ei] = encoder_hidden[-1, 0]

        return encoder_hidden, encoder_hidden_all


class DecoderCell(nn.Module):
    def __init__(self, output_size, hidden_size, n_layers=1, rnn_type='RNN', dropout_p=0.1, device='cpu'):
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


class AttnDecoderCell(nn.Module):
    def __init__(self, output_size, hidden_size, n_layers=1, rnn_type='RNN', dropout_p=0.1, device='cpu'):
        super(AttnDecoderCell, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.output_size = output_size
        self.RNN_type = rnn_type

        self.rnn = nn.__dict__[self.RNN_type](
            input_size=self.hidden_size,
            hidden_size=self.hidden_size * 2,
            num_layers=self.n_layers,
            dropout=dropout_p
        )
        self.W = nn.Parameter(torch.randn((self.hidden_size, self.hidden_size), device=device))
        self.U = nn.Parameter(torch.randn((self.hidden_size, self.hidden_size), device=device))
        self.v = nn.Parameter(torch.randn((self.hidden_size, 1), device=device))

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        self.dropout = nn.Dropout(dropout_p)

        self.out = nn.Linear(self.hidden_size * 2, output_size)

    def e(self, g, h):
        """Computes the similarity between the previous decoder hidden state g and an encoder hidden state h"""
        # vT tanh(W g_(i-1) + U h_t)
        return self.v.T @ torch.tanh(self.W * g + self.U * h)

    def alpha(self, encoder_hiddens, input_hidden, t):
        """Computes the attention weight for a given encoder hidden state"""
        # alpha_it = exp(e(g_(i-1), h_t)) / sum(exp(e(g_(i-1), h_j)))
        T = len(encoder_hiddens)
        numerator = torch.exp(self.e(input_hidden, encoder_hiddens[t]))

        denominator = 0

        for j in range(T):
            denominator += torch.exp(self.e(input_hidden, encoder_hiddens[j]))

        return numerator / denominator

    def forward(self, input, input_hidden, encoder_hiddens):

        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # c_i = sum(alpha_it * h_t)
        c_i = 0

        for t in range(len(encoder_hiddens)):
            alpha_it = self.alpha(encoder_hiddens, input_hidden, t)
            h_t = encoder_hiddens[t]
            c_i += alpha_it * h_t

        hidden = torch.concat((input_hidden, c_i), dim=2)  # Concatenate the context vector and the decoder hidden state

        output, hidden = self.rnn(embedded, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)

        # Seperate the concatenated hidden state into the decoder hidden state and the context vector
        hidden, context = torch.split(hidden, self.hidden_size, dim=2)

        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, n_layers=1, rnn_type='RNN', dropout_p=0.1, attention=False,
                 device='cpu'):
        super(DecoderRNN, self).__init__()

        self.attention = attention
        if attention:
            self.decoder_cell = AttnDecoderCell(output_size, hidden_size, n_layers, rnn_type, dropout_p, device)
        else:
            self.decoder_cell = DecoderCell(output_size, hidden_size, n_layers, rnn_type, dropout_p, device)

    def forward(self, input, hidden, encoder_hiddens=None):
        assert encoder_hiddens is not None if self.attention else True  # If attention is used, all encoder hidden states must be provided
        if self.attention:
            output, hidden = self.decoder_cell(input, hidden, encoder_hiddens)
        else:
            output, hidden = self.decoder_cell(input, hidden)

        return output, hidden
