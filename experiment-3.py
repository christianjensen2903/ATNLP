from collections import defaultdict

from torch.utils.data import DataLoader

import scan_dataset
import models
import pipeline
import torch
import wandb
import os
from matplotlib import pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm

log_wandb = False

n_iter = 100000
n_runs = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


input_lang = scan_dataset.Lang()
output_lang = scan_dataset.Lang()

train_dataset = scan_dataset.ScanDataset(
    split=scan_dataset.ScanSplit.ADD_PRIM_SPLIT,
    input_lang=input_lang,
    output_lang=output_lang,
    train=True,
    split_variation='turn_left'
)

test_dataset = scan_dataset.ScanDataset(
    split=scan_dataset.ScanSplit.ADD_PRIM_SPLIT,
    input_lang=input_lang,
    output_lang=output_lang,
    train=False,
    split_variation='turn_left'
)

MAX_LENGTH = max(train_dataset.input_lang.max_length, train_dataset.output_lang.max_length)
max_length=MAX_LENGTH



overall_best = {
    'HIDDEN_SIZE': 200,  # 25, 50, 100, 200, or 400
    'RNN_TYPE': 'LSTM',  # RNN, GRU or LSTM
    'N_LAYERS': 2,  # 1 or 2
    'DROPOUT': 0.5,  # 0, 0.1 or 0.5
    'ATTENTION': False,  # True or False
}

experiment_best = {
    'HIDDEN_SIZE': 100,  # 25, 50, 100, 200, or 400
    'RNN_TYPE': 'GRU',  # RNN, GRU or LSTM
    'N_LAYERS': 1,  # 1 or 2
    'DROPOUT': 0.5,  # 0, 0.1 or 0.5
    'ATTENTION': True,  # True or False
}



def run_overall_best():
    results = []
    # Train 5 times and average the results
    for run in range(n_runs):
        encoder = models.EncoderRNN(train_dataset.input_lang.n_words, overall_best['HIDDEN_SIZE'], MAX_LENGTH, device,
                                    overall_best['N_LAYERS'], overall_best['RNN_TYPE'], overall_best['DROPOUT']).to(
            device)
        decoder = models.DecoderRNN(train_dataset.output_lang.n_words, overall_best['HIDDEN_SIZE'],
                                    overall_best['N_LAYERS'], overall_best['RNN_TYPE'], overall_best['DROPOUT'],
                                    overall_best['ATTENTION']).to(device)

        encoder, decoder = pipeline.train(train_dataset, encoder, decoder, n_iter, print_every=100, learning_rate=0.001,
                                          device=device, log_wandb=log_wandb)
        pickle.dump(encoder, open(f'runs/overall_best_encoder_exp_2_run_{run}.sav', 'wb'))
        pickle.dump(decoder, open(f'runs/overall_best_decoder_exp_2_run_{run}.sav', 'wb'))
        results.append(pipeline.evaluate(test_dataset, encoder, decoder, max_length=MAX_LENGTH, verbose=False))

    avg_accuracy = sum(results) / len(results)
    print('Average accuracy for overall best: {}'.format(avg_accuracy))
    if log_wandb:
        wandb.run.summary["Average accuracy for overall best"] = avg_accuracy


def run_experiment_best():
    results = []
    # Train 5 times and average the results
    for run in range(n_runs):
        encoder = models.EncoderRNN(train_dataset.input_lang.n_words, experiment_best['HIDDEN_SIZE'], MAX_LENGTH, device,
                                    experiment_best['N_LAYERS'], experiment_best['RNN_TYPE'],
                                    experiment_best['DROPOUT']).to(device)
        decoder = models.DecoderRNN(train_dataset.output_lang.n_words, experiment_best['HIDDEN_SIZE'],
                                    experiment_best['N_LAYERS'], experiment_best['RNN_TYPE'],
                                    experiment_best['DROPOUT'],
                                    experiment_best['ATTENTION']).to(device)

        print(encoder)
        print(decoder)

        encoder, decoder = pipeline.train(train_dataset, encoder, decoder, n_iter, print_every=100, learning_rate=0.001,
                                          device=device, log_wandb=log_wandb)
        pickle.dump(encoder, open(f'runs/experiment_best_encoder_exp_3_run_{run}.sav', 'wb'))
        pickle.dump(decoder, open(f'runs/experiment_best_decoder_exp_3_run_{run}.sav', 'wb'))
        acc = pipeline.evaluate(test_dataset, encoder, decoder, max_length=MAX_LENGTH, verbose=False)
        results.append(acc)
        print(f'Accuracy for run {run}: {acc}')
        
    avg_accuracy = sum(results) / len(results)
    print('Average accuracy for experiment best: {}'.format(avg_accuracy))
    if log_wandb:
        wandb.run.summary["Average accuracy for experiment best"] = avg_accuracy


def main():
    # WANDB_API_KEY = os.environ.get('WANDB_API_KEY')
    if log_wandb:
        wandb.login()
        wandb.init(project="experiment-3", entity="atnlp")

    # run_overall_best()
    run_experiment_best()
    # test_sequence_length()
    # test_command_length()

    # inspect_greedy_search()
    # oracle_test()
    # test_sequence_length(oracle=True)


if __name__ == '__main__':
    main()
