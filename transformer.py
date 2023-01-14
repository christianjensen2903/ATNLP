from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
import torch.nn.functional as F
from Seq2SeqModel import Seq2SeqModel, Seq2SeqModelConfig
from dataclasses import dataclass

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


@dataclass
class Seq2SeqTransformerConfig(Seq2SeqModelConfig):
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    emb_size: int = 512
    nhead: int = 8
    input_vocab_size: int = 1000
    target_vocab_size: int = 1000
    dim_feedforward: int = 512
    dropout: float = 0.1


# Seq2Seq Network
class Seq2SeqTransformer(Seq2SeqModel):
    def __init__(
        self,
        config: Seq2SeqTransformerConfig,
    ):
        super(Seq2SeqTransformer, self).__init__(config=config)
        self.config = config
        if config is not None:
            self.from_config(config)

    def from_config(self, config: Seq2SeqTransformerConfig):

        self.transformer = Transformer(
            d_model=config.emb_size,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.generator = nn.Linear(config.emb_size, config.target_vocab_size)
        self.input_tok_emb = TokenEmbedding(config.input_vocab_size, config.emb_size)
        self.target_tok_emb = TokenEmbedding(config.target_vocab_size, config.emb_size)
        self.positional_encoding = PositionalEncoding(
            config.emb_size, dropout=config.dropout
        )
        self.pad_index = config.pad_index
        self.eos_index = config.eos_index
        self.sos_index = config.sos_index

        return self

    def reset_model(self):
        self.from_config(self.config)

        return self

    def forward(self, input: Tensor, target: Tensor):

        # Get masks for input and target
        (
            input_mask,
            target_mask,
            input_padding_mask,
            target_padding_mask,
        ) = self.create_mask(input, target)

        # Embed sequence
        input = self.input_tok_emb(input)
        target = self.target_tok_emb(target)

        # Add positional encoding
        input = self.positional_encoding(input)
        target = self.positional_encoding(target)

        # print(target_mask)

        # Pass through transformer
        outs = self.transformer(
            input,
            target,
            input_mask,
            target_mask,
            None,
            input_padding_mask,
            target_padding_mask,
        )

        # Generate output
        outs = self.generator(outs)
        # return F.log_softmax(outs, dim=1)
        return outs

    def encode(self, input: Tensor, input_mask: Tensor):
        return self.transformer.encoder(
            self.positional_encoding(self.input_tok_emb(input)), input_mask
        )

    def decode(self, target: Tensor, memory: Tensor, target_mask: Tensor):
        return self.transformer.decoder(
            self.positional_encoding(self.target_tok_emb(target)), memory, target_mask
        )

    def create_mask(self, input, target):

        input_seq_len = input.shape[1]
        target_seq_len = target.shape[1]

        target_mask = self.transformer.generate_square_subsequent_mask(target_seq_len)
        input_mask = torch.zeros((input_seq_len, input_seq_len)).type(torch.bool)

        input_padding_mask = input == self.pad_index
        target_padding_mask = target == self.pad_index

        return input_mask, target_mask, input_padding_mask, target_padding_mask

    def predict(
        self,
        input: torch.Tensor,
        max_length: int = 100,
        oracle_length: int = None,
        oracle_target: torch.Tensor = None,
    ):
        preds = torch.ones(1, 1).fill_(self.sos_index).type(torch.long)

        for t in range(1, max_length):
            out = self.forward(input, preds)
            print(out.shape)
            pred = out.argmax(2)[:, -1].unsqueeze(1)

            preds = torch.cat([preds, pred], dim=1)

            if pred == self.eos_index:
                break

        return preds, None

        # num_tokens = input.shape[1]
        # src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        # tgt_tokens = self.greedy_decode(
        #     input, src_mask, max_len=num_tokens + 5, start_symbol=self.sos_index
        # ).flatten()
        # return tgt_tokens, None

    def greedy_decode(self, src, src_mask, max_len, start_symbol):

        src = src
        src_mask = src_mask

        memory = self.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long)
        for i in range(max_len - 1):
            memory = memory
            tgt_mask = self.transformer.generate_square_subsequent_mask(
                ys.size(0)
            ).type(torch.bool)
            print(ys.shape)
            out = self.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = self.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat(
                [ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0
            )
            if next_word == self.eos_index:
                break
        return ys
