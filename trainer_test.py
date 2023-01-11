from collections import defaultdict

import scan_dataset_new
import models_new
import pipeline_new
import torch
import wandb
import os
from matplotlib import pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm
import Seq2SeqTrainer

input_lang = scan_dataset_new.Lang()
output_lang = scan_dataset_new.Lang()

train_dataset = scan_dataset_new.ScanDataset(
    split=scan_dataset_new.ScanSplit.LENGTH_SPLIT,
    input_lang=input_lang,
    output_lang=output_lang,
    train=True
)

test_dataset = scan_dataset_new.ScanDataset(
    split=scan_dataset_new.ScanSplit.LENGTH_SPLIT,
    input_lang=input_lang,
    output_lang=output_lang,
    train=False
)

n_iter = 10000
n_runs = 1


overall_best = {
    'HIDDEN_SIZE': 200,  # 25, 50, 100, 200, or 400
    'RNN_TYPE': 'LSTM',  # RNN, GRU or LSTM
    'N_LAYERS': 2,  # 1 or 2
    'DROPOUT': 0.5,  # 0, 0.1 or 0.5
    'ATTENTION': False,  # True or False
}

config = models_new.RNNSeq2SeqConfig(
    pad_index=train_dataset.input_lang.pad_index,
    sos_index=train_dataset.input_lang.sos_index,
    eos_index=train_dataset.input_lang.eos_index,
    hidden_size=overall_best['HIDDEN_SIZE'],
    n_layers=overall_best['N_LAYERS'],
    dropout_p=overall_best['DROPOUT'],
    attention=overall_best['ATTENTION'],
    rnn_type=overall_best['RNN_TYPE'],
    teacher_forcing_ratio=0.5,
    input_vocab_size=train_dataset.input_lang.n_words,
    output_vocab_size=train_dataset.output_lang.n_words,
    )
model = models_new.RNNSeq2Seq().from_config(config)

train_args = Seq2SeqTrainer.Seq2SeqTrainingArguments(
    batch_size=1,
    n_iter=n_iter,
    learning_rate=0.001,
    print_every=1000,
    clip_grad=5.0,
    output_dir='checkpoints',
)

trainer = Seq2SeqTrainer.Seq2SeqTrainer(
    model=model,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    args=train_args,
)

model = trainer.train(verbose=True, evaluate_after=False)


# model = pipeline_new.train(train_dataset, model, n_iter, print_every=1000, learning_rate=0.001, verbose=True)
pipeline_new.evaluate(test_dataset, model)