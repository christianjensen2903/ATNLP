from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
import torch.nn.functional as F
from Seq2SeqModel import Seq2SeqModel, Seq2SeqModelConfig
from dataclasses import dataclass
from transformers import MarianMTModel, MarianConfig


@dataclass
class Seq2SeqTransformerConfig(Seq2SeqModelConfig):
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    emb_size: int = 512
    nhead: int = 8
    dim_feedforward: int = 512
    dropout: float = 0.1


# Seq2Seq Network
class Seq2SeqTransformer(Seq2SeqModel):
    def __init__(
        self,
        config: Seq2SeqTransformerConfig = None,
    ):
        super(Seq2SeqTransformer, self).__init__(config=config)
        self.config = config
        if config is not None:
            self.from_config(config)

    def from_config(self, config: Seq2SeqTransformerConfig):
        self.config = config
        marianConfig = MarianConfig(
            vocab_size=config.input_vocab_size,
            decoder_vocab_size=config.output_vocab_size,
            d_model=config.emb_size,
            encoder_layers=config.num_encoder_layers,
            decoder_layers=config.num_decoder_layers,
            encoder_attention_heads=config.nhead,
            decoder_attention_heads=config.nhead,
            encoder_ffn_dim=config.dim_feedforward,
            decoder_ffn_dim=config.dim_feedforward,
            dropout=config.dropout,
            pad_token_id=config.pad_index,
            eos_token_id=config.eos_index,
            max_position_embeddings=64,
            decoder_start_token_id=config.sos_index,
            share_encoder_decoder_embeddings=False,
            tie_word_embeddings=False,
        )

        self.transformer = MarianMTModel(marianConfig)
        return self

    def reset_model(self):
        self.from_config(self.config)
        return self

    def forward(self, input: Tensor, target: Tensor):
        return self.transformer(input_ids=input, labels=target).logits

    def predict(
        self,
        input: torch.Tensor,
        max_length: int = 100,
        oracle_length: int = None,
        oracle_target: torch.Tensor = None,
    ):

        min_length = 1
        # If oracle is provided, use it as the max length
        if oracle_length is not None:
            max_length = oracle_length
            min_length = oracle_length

        # TODO: Isn't using the oracle target properly
        if oracle_target is not None:
            max_length = oracle_target.size(1)
            min_length = oracle_target.size(1)

        oracle_target = oracle_target.squeeze() if oracle_target is not None else None

        output = self.transformer.generate(
            input_ids=input,
            max_length=max_length
            + 1,  # +1 due to the model generating an extra token of some reason,
            min_length=min_length,
            bos_token_id=self.config.sos_index,
            pad_token_id=self.config.pad_index,
            eos_token_id=self.config.eos_index,
        )
        output = output[:, 1:]  # remove the extra token
        return output, 0  # TODO: return the probability
        # return self.transformer.generate(input_ids=input, max_length=max_length), None
