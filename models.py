import torch
import torch.nn as nn
import torch.nn.functional as F

def init_hidden(rnn_type, n_layers, hidden_size, device):
    if rnn_type == 'LSTM':
        return (
            torch.zeros(n_layers, 1, hidden_size, device=device),
            torch.zeros(n_layers, 1, hidden_size, device=device)
        )
    return torch.zeros(n_layers, 1, hidden_size, device=device)

class EncoderRNN(nn.Module):
    def __init__(self, input_size, config):
        super(EncoderRNN, self).__init__()
        self.hidden_size = config['HIDDEN_SIZE']
        self.n_layers = config['N_LAYERS']

        self.embedding = nn.Embedding(input_size, self.hidden_size)

        self.dropout = nn.Dropout(config['DROPOUT'])

        self.RNN_type = config['RNN_TYPE']

        self.rnn = nn.__dict__[self.RNN_type](
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            dropout=config['DROPOUT']
        )

    def forward(self, encoder_input, hidden):
        output = self.embedding(encoder_input).view(1, 1, -1)
        output = self.dropout(output)
        output, hidden = self.rnn(output, hidden)
        return output, hidden

    def init_hidden(self, device):
        return init_hidden(self.RNN_type, self.n_layers, self.hidden_size, device)


class DecoderRNN(nn.Module):
    def __init__(self, output_size, config):
        super(DecoderRNN, self).__init__()
        self.hidden_size = config['HIDDEN_SIZE']
        self.n_layers = config['N_LAYERS']

        self.embedding = nn.Embedding(output_size, self.hidden_size)

        self.dropout = nn.Dropout(config['DROPOUT'])

        self.RNN_type = config['RNN_TYPE']

        self.rnn = nn.__dict__[self.RNN_type](
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            dropout=config['DROPOUT']
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

    def init_hidden(self):
        return init_hidden(self.RNN_type, self.n_layers, self.hidden_size)