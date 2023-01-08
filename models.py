import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn
import random

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, rnn_type='RNN', dropout_p=0.1, device='cpu'):
        super(EncoderRNN, self).__init__()
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
            dropout=dropout_p,
            batch_first=True
        )

    def forward(self, input):
        input = self.embedding(input)
        input = self.dropout(input)
        output, hidden = self.rnn(input)

        # last layer hidden state, all hidden state
        return hidden, output


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
            dropout=dropout_p,
            batch_first=True
        )

        self.out = nn.Linear(self.hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, decoder_input, hidden):
        output = self.embedding(decoder_input)
        output = self.dropout(output)

        output, hidden = self.rnn(output, hidden)

        output = self.out(output[:, -1, :])
        output = self.softmax(output)
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
        self.attention_weights = F.softmax(scores, dim=0)

        bmm = torch.bmm(self.attention_weights, values)
        return torch.sum(bmm, dim=0)

class AttnDecoderCell(nn.Module):
    def __init__(self, ouput_size, hidden_size, num_layers, rnn_type,
                 dropout_p=0, device='cpu', max_length=100):
        super().__init__()
        self.attention = AdditiveAttention(num_hiddens=hidden_size, key_size=hidden_size, query_size=hidden_size)
        self.embedding = nn.Embedding(ouput_size, hidden_size)
        self.rnn = nn.GRU(
            hidden_size*2, hidden_size, num_layers,
            dropout=dropout_p,
            batch_first=True)
        self.dense = nn.Linear(hidden_size*2, ouput_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input, hidden_state, enc_outputs):
        
        # TODO: Doesn't handle batch size > 1
        # Reshaping to handle sequence length as batch to use bmm
        enc_outputs = enc_outputs.permute(1, 0, 2)

        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        # TODO: Doesn't handle LSTM
        # Only using last layer hidden state for attention
        query = torch.unsqueeze(hidden_state[-1], dim=1)
        context = self.attention(
            query, enc_outputs, enc_outputs)

        context = context.unsqueeze(0)

        x = torch.cat((context, embedded), dim=-1)

        outputs, hidden_state = self.rnn(x, hidden_state)

        # Concat context and and hidden state to get output
        query = torch.unsqueeze(hidden_state[-1], dim=1)
        x = torch.cat((context, query), dim=-1)
        outputs = F.log_softmax(self.dense(x[0]), dim=1)

        return outputs, hidden_state


class DecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, n_layers=1, rnn_type='RNN', dropout_p=0.1, attention=False,
                 device='cpu', max_length=100):
        super(DecoderRNN, self).__init__()

        self.output_size = output_size

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


# Abstract class for seq2seq models
class Seq2SeqModel(nn.Module):
    def __init__(self, pad_index, sos_index, eos_index, **kwargs):
        super(Seq2SeqModel, self).__init__()
        self.pad_index = pad_index
        self.sos_index = sos_index
        self.eos_index = eos_index

    def forward(self, input, target):
        raise NotImplementedError

    def predict(self, input, max_length=100):
        raise NotImplementedError


class RNNSeq2Seq(Seq2SeqModel):
    def __init__(self, pad_index, sos_index, eos_index, encoder, decoder, teacher_forcing_ratio=0.5, device='cpu'):
        super(RNNSeq2Seq, self).__init__(pad_index, sos_index, eos_index)
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, input, target):
        batch_size = input.size(0)
        max_len = target.size(1)
        vocab_size = self.decoder.output_size

        # Initialize the output sequence with the SOS token
        outputs = torch.zeros(batch_size, max_len, vocab_size).to(self.device)
        outputs[:, 0] = self.sos_index

        hidden, enc_outputs = self.encoder(input)
        decoder_input = torch.full((batch_size, 1), self.sos_index, dtype=torch.long).to(self.device)

        for t in range(1, max_len):
            # Teacher forcing: Feed the target as the next input
            output, hidden = self.decoder(decoder_input, hidden, enc_outputs)
            outputs[:, t] = output
            teacher_force = random.random() < self.teacher_forcing_ratio
            top1 = output.max(1)[1]
            decoder_input = (target[:, t] if teacher_force else top1).unsqueeze(1)

        return outputs

    def predict(self, input, max_length=100):
        """Predict the output sequence given the input sequence using greedy search."""
        batch_size = input.size(0)

        # Initialize the output sequence with the SOS token
        outputs = torch.full((batch_size, max_length), self.pad_index, dtype=torch.long).to(self.device)
        outputs[:, 0] = self.sos_index

        hidden, enc_outputs = self.encoder(input)
        decoder_input = torch.full((batch_size, 1), self.sos_index, dtype=torch.long).to(self.device)

        # Decode the sequence one timestep at a time
        for t in range(1, max_length):
            output, hidden = self.decoder(decoder_input, hidden, enc_outputs)

            outputs[:, t] = output.max(1)[1]

            # Find the indices of the EOS tokens in the output
            eos_indices = torch.where(output.max(1)[1] == self.eos_index)

            # If EOS is found set the rest of the sequence to PAD
            if len(eos_indices[0]) > 0:
                outputs[eos_indices[0], t+1:max_length] = self.pad_index
                
            decoder_input = output.max(1)[1].unsqueeze(1)

        return outputs
