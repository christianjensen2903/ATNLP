import scan_dataset
import models
import pipeline
import torch
from matplotlib import pyplot as plt
import wandb
import numpy as np
from transformers import MarianMTModel, MarianTokenizer
from transformers import MarianConfig
import torch
from transformer import Seq2SeqTransformer


input_lang = scan_dataset.Lang()
output_lang = scan_dataset.Lang()

train_dataset = scan_dataset.ScanDataset(
    split=scan_dataset.ScanSplit.SIMPLE_SPLIT,
    input_lang=input_lang,
    output_lang=output_lang,
    train=True
)

test_dataset = scan_dataset.ScanDataset(
    split=scan_dataset.ScanSplit.SIMPLE_SPLIT,
    input_lang=input_lang,
    output_lang=output_lang,
    train=False
)

MAX_LENGTH = max(train_dataset.input_lang.max_length, train_dataset.output_lang.max_length)


SRC_VOCAB_SIZE = input_lang.n_words
TGT_VOCAB_SIZE = output_lang.n_words
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3


transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

# # Sample a random sentence from the training set
X, y = train_dataset[0]

# Convert to torch tensor
X, y = train_dataset.convert_to_tensor(X, y)

print(transformer(X, y))




# lang = scan_dataset.Lang()

# train_dataset = scan_dataset.ScanDataset(
#     split=scan_dataset.ScanSplit.SIMPLE_SPLIT,
#     input_lang=lang,
#     output_lang=lang,
#     train=True
# )

# # Initialize a model without pretrained weights
# config = MarianConfig(
#     vocab_size=lang.n_words,
#     pad_token_id=scan_dataset.PAD_token,
#     eos_token_id=scan_dataset.EOS_token,
#     forced_eos_token_id=scan_dataset.EOS_token,
#     d_model=16,
#     )
# model = MarianMTModel(config=config)

# # # Sample a random sentence from the training set
# X, y = train_dataset[0]

# # Convert to torch tensor
# X, y = train_dataset.convert_to_tensor(X, y)


# # pass input_ids to encoder and to decoder and pass BOS token to decoder to retrieve first logit
# outputs = model(X, decoder_input_ids=y, return_dict=True)

# # get logits
# lm_logits = outputs.logits

# print(lm_logits)


