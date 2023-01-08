from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 input_vocab_size: int,
                 target_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       batch_first=True)
        self.generator = nn.Linear(emb_size, target_vocab_size)
        self.input_tok_emb = TokenEmbedding(input_vocab_size, emb_size)
        self.target_tok_emb = TokenEmbedding(target_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                input: Tensor,
                target: Tensor):

        # Get masks for input and target
        input_mask, target_mask, input_padding_mask, target_padding_mask = self.create_mask(input, target)
        
        # Embed sequence
        input = self.input_tok_emb(input)
        target = self.target_tok_emb(target)

        # Add positional encoding
        input = self.positional_encoding(input)
        target = self.positional_encoding(target)

        # Pass through transformer
        outs = self.transformer(input, target, input_mask, target_mask, None,
                                input_padding_mask, target_padding_mask)

        # Generate output
        return self.generator(outs)

    def encode(self, input: Tensor, input_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.input_tok_emb(input)), input_mask)

    def decode(self, target: Tensor, memory: Tensor, target_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.target_tok_emb(target)), memory,
                          target_mask)


    def create_mask(self, input, target):
        input_seq_len = input.shape[1]
        target_seq_len = target.shape[1]

        target_mask = self.transformer.generate_square_subsequent_mask(target_seq_len)
        input_mask = torch.zeros((input_seq_len, input_seq_len)).type(torch.bool)

        input_padding_mask = (input == self.pad_idx)
        target_padding_mask = (target == self.pad_idx)
        return input_mask, target_mask, input_padding_mask, target_padding_mask