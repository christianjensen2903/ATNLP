import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from enum import Enum
import re
import random
import wandb
from tqdm import tqdm
import helper
import time
import scan_dataset
import models
import pipeline


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

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

MAX_LENGTH = max(train_dataset.input_lang.max_length, train_dataset.output_lang.max_length)


config = {
    'HIDDEN_SIZE': 256, # 25, 50, 100, 200, or 400
    'RNN_TYPE': 'RNN', # RNN, GRU or LSTM
    'N_LAYERS': 2, # 1 or 2
    'DROPOUT': 0, # 0, 0.1 or 0.5
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = models.EncoderRNN(train_dataset.input_lang.n_words, config['HIDDEN_SIZE'], device, config['N_LAYERS'], config['RNN_TYPE'], config['DROPOUT']).to(device)
decoder = models.DecoderRNN(train_dataset.output_lang.n_words, config['HIDDEN_SIZE'], config['N_LAYERS'], config['RNN_TYPE'], config['DROPOUT']).to(device)
# decoder1 = AttnDecoderRNN(train_dataset.output_lang.n_words, config['HIDDEN_SIZE'], config['N_LAYERS'], config['RNN_TYPE'], config['RNN_TYPE']).to(device)

encoder, decoder = pipeline.train(encoder, decoder, 10000, print_every=100, learning_rate=0.001)

pipeline.evaluate(test_dataset, encoder, decoder, max_length=100, verbose=True)

