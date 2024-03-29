import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import random
from Seq2SeqModel import Seq2SeqModel, Seq2SeqModelConfig
from dataclasses import dataclass


class EncoderRNN(nn.Module):
    def __init__(
        self, input_size, hidden_size, n_layers=1, rnn_type="RNN", dropout_p=0.1
    ):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.RNN_type = rnn_type

        self.embedding = nn.Embedding(input_size, self.hidden_size)

        self.dropout = nn.Dropout(dropout_p)

        self.rnn = nn.__dict__[self.RNN_type](
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            dropout=dropout_p,
            batch_first=True,
        )

    def forward(self, input):
        input = self.embedding(input)
        input = self.dropout(input)
        output, hidden = self.rnn(input)

        # last layer hidden state, all hidden state
        return hidden, output

    def reset_model(self):
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

        return self


class DecoderCell(nn.Module):
    def __init__(
        self,
        output_size,
        hidden_size,
        n_layers=1,
        rnn_type="RNN",
        dropout_p=0.1,
        max_length=100,
    ):
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
            batch_first=True,
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
    def __init__(
        self, ouput_size, hidden_size, num_layers, rnn_type, dropout_p=0, max_length=100
    ):
        super().__init__()
        self.attention = AdditiveAttention(
            num_hiddens=hidden_size, key_size=hidden_size, query_size=hidden_size
        )
        self.embedding = nn.Embedding(ouput_size, hidden_size)
        self.rnn = nn.GRU(
            hidden_size * 2,
            hidden_size,
            num_layers,
            dropout=dropout_p,
            batch_first=True,
        )
        self.dense = nn.Linear(hidden_size * 2, ouput_size)
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
        context = self.attention(query, enc_outputs, enc_outputs)

        context = context.unsqueeze(0)

        x = torch.cat((context, embedded), dim=-1)

        outputs, hidden_state = self.rnn(x, hidden_state)

        # Concat context and and hidden state to get output
        query = torch.unsqueeze(hidden_state[-1], dim=1)
        x = torch.cat((context, query), dim=-1)
        outputs = F.log_softmax(self.dense(x[0]), dim=1)

        return outputs, hidden_state


class DecoderRNN(nn.Module):
    def __init__(
        self,
        output_size,
        hidden_size,
        n_layers=1,
        rnn_type="RNN",
        dropout_p=0.1,
        attention=False,
        max_length=100,
    ):
        super(DecoderRNN, self).__init__()

        self.output_size = output_size

        self.attention = attention
        if attention:
            self.decoder_cell = AttnDecoderCell(
                output_size, hidden_size, n_layers, rnn_type, dropout_p, max_length
            )
        else:
            self.decoder_cell = DecoderCell(
                output_size, hidden_size, n_layers, rnn_type, dropout_p
            )

    def forward(self, input, hidden, enc_outputs=None):
        assert (
            enc_outputs is not None if self.attention else True
        )  # If attention is used, all encoder hidden states must be provided
        if self.attention:
            output, hidden = self.decoder_cell(input, hidden, enc_outputs)
        else:
            output, hidden = self.decoder_cell(input, hidden)

        return output, hidden

    def reset_model(self):
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

        return self


@dataclass
class RNNSeq2SeqConfig(Seq2SeqModelConfig):
    teacher_forcing_ratio: float = 0.5
    hidden_size: int = 256
    n_layers: int = 1
    dropout_p: float = 0.1
    rnn_type: str = "RNN"
    attention: bool = False
    input_vocab_size: int = 0
    output_vocab_size: int = 0


class RNNSeq2Seq(Seq2SeqModel):
    def __init__(self, config: RNNSeq2SeqConfig = None):
        super(RNNSeq2Seq, self).__init__(config)
        self.config = config
        if config:
            self.from_config(config)

    def from_config(self, config: RNNSeq2SeqConfig):
        self.pad_index = config.pad_index
        self.sos_index = config.sos_index
        self.eos_index = config.eos_index
        self.input_vocab_size = config.input_vocab_size
        self.output_vocab_size = config.output_vocab_size
        self.hidden_size = config.hidden_size
        self.n_layers = config.n_layers
        self.dropout_p = config.dropout_p
        self.rnn_type = config.rnn_type
        self.attention = config.attention
        self.teacher_forcing_ratio = config.teacher_forcing_ratio
        self.encoder = EncoderRNN(
            self.input_vocab_size,
            self.hidden_size,
            self.n_layers,
            self.rnn_type,
            self.dropout_p,
        )
        self.decoder = DecoderRNN(
            self.output_vocab_size,
            self.hidden_size,
            self.n_layers,
            self.rnn_type,
            self.dropout_p,
            self.attention,
        )

        self.reset_model()

        return self

    def reset_model(self):
        self.encoder.reset_model()
        self.decoder.reset_model()

        return self

    def forward(self, input, target):
        batch_size = input.size(0)
        max_len = target.size(1)
        vocab_size = self.decoder.output_size

        # Initialize the output sequence with the SOS token
        outputs = torch.zeros(batch_size, max_len, vocab_size)
        outputs[:, 0] = self.sos_index

        hidden, enc_outputs = self.encoder(input)
        decoder_input = torch.full((batch_size, 1), self.sos_index, dtype=torch.long)

        for t in range(1, max_len):
            # Teacher forcing: Feed the target as the next input
            output, hidden = self.decoder(decoder_input, hidden, enc_outputs)
            outputs[:, t] = output
            teacher_force = random.random() < self.teacher_forcing_ratio
            top1 = output.max(1)[1]
            decoder_input = (target[:, t] if teacher_force else top1).unsqueeze(1)

        return outputs

    def predict(self, input, max_length=60, oracle_length=None, oracle_target=None):

        # If oracle is provided, use it as the max length
        if oracle_length is not None:
            max_length = oracle_length

        if oracle_target is not None:
            max_length = oracle_target.size(1)

        oracle_target = oracle_target.squeeze() if oracle_target is not None else None

        # Initialize the output sequence with the SOS token
        outputs = torch.full((max_length,), self.pad_index, dtype=torch.long)
        outputs[0] = self.sos_index

        hidden, enc_outputs = self.encoder(input)
        decoder_input = torch.full((1, 1), self.pad_index, dtype=torch.long)

        log_prob = 0

        # Decode the sequence one timestep at a time
        for t in range(1, max_length):
            output, hidden = self.decoder(decoder_input, hidden, enc_outputs)

            if oracle_target is not None:
                log_prob += output.squeeze().detach()[oracle_target[t]].item()
                outputs[t] = oracle_target[t]
            else:
                # Take the most likely word index (highest value) from the output
                log_prob += output.squeeze().detach().max().item()
                outputs[t] = output.squeeze().argmax()

                if oracle_length is not None:
                    n_tokens = 4  # Number of tokens to consider
                    # If eos is found but not at the end of the sequence take the next most likely word
                    if outputs[t].item() < n_tokens and t < max_length - 1:
                        # print(output.squeeze()[n_tokens:].argmax() + n_tokens)
                        outputs[t] = output.squeeze()[n_tokens:].argmax() + n_tokens
                else:
                    # If the predicted word is EOS, stop predicting
                    if outputs[t] == self.eos_index:
                        break

            decoder_input = outputs[t].unsqueeze(0).unsqueeze(0)

        # If oracle length set last output to EOS
        if oracle_length is not None:
            outputs[oracle_length - 1] = self.eos_index

        return outputs, log_prob
