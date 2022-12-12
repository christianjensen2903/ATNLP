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

wandb_log = True

input_lang = scan_dataset.Lang()
output_lang = scan_dataset.Lang()

train_dataset = scan_dataset.ScanDataset(
    split=scan_dataset.ScanSplit.LENGTH_SPLIT,
    input_lang=input_lang,
    output_lang=output_lang,
    train=True
)

test_dataset = scan_dataset.ScanDataset(
    split=scan_dataset.ScanSplit.LENGTH_SPLIT,
    input_lang=input_lang,
    output_lang=output_lang,
    train=False
)

MAX_LENGTH = max(train_dataset.input_lang.max_length, train_dataset.output_lang.max_length)

n_iter = 100000
n_runs = 5

overall_best = {
    'HIDDEN_SIZE': 200,  # 25, 50, 100, 200, or 400
    'RNN_TYPE': 'LSTM',  # RNN, GRU or LSTM
    'N_LAYERS': 2,  # 1 or 2
    'DROPOUT': 0.5,  # 0, 0.1 or 0.5
    'ATTENTION': False,  # True or False
}

experiment_best = {
    'HIDDEN_SIZE': 50,  # 25, 50, 100, 200, or 400
    'RNN_TYPE': 'GRU',  # RNN, GRU or LSTM
    'N_LAYERS': 1,  # 1 or 2
    'DROPOUT': 0.5,  # 0, 0.1 or 0.5
    'ATTENTION': True,  # True or False
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


def run_overall_best():
    results = []
    # Train 5 times and average the results
    for _ in range(n_runs):
        encoder = models.EncoderRNN(train_dataset.input_lang.n_words, overall_best['HIDDEN_SIZE'], device,
                                    overall_best['N_LAYERS'], overall_best['RNN_TYPE'], overall_best['DROPOUT']).to(
            device)
        decoder = models.DecoderRNN(train_dataset.output_lang.n_words, overall_best['HIDDEN_SIZE'],
                                    overall_best['N_LAYERS'], overall_best['RNN_TYPE'], overall_best['DROPOUT'],
                                    overall_best['ATTENTION']).to(device)

        encoder, decoder = pipeline.train(train_dataset, encoder, decoder, n_iter, print_every=100, learning_rate=0.001,
                                          device=device)
        pickle.dump(encoder, open('overall_best_encoder_exp_2.sav', 'wb'))
        pickle.dump(decoder, open('overall_best_decoder_exp_2.sav', 'wb'))
        results.append(pipeline.evaluate(test_dataset, encoder, decoder, max_length=MAX_LENGTH, verbose=False))

    avg_accuracy = sum(results) / len(results)
    print('Average accuracy for overall best: {}'.format(avg_accuracy))
    if wandb_log:
        wandb.run.summary["Average accuracy for overall best"] = avg_accuracy


def run_experiment_best():
    results = []
    # Train 5 times and average the results
    for _ in range(n_runs):
        encoder = models.EncoderRNN(train_dataset.input_lang.n_words, experiment_best['HIDDEN_SIZE'], device,
                                    experiment_best['N_LAYERS'], experiment_best['RNN_TYPE'],
                                    experiment_best['DROPOUT']).to(device)
        decoder = models.DecoderRNN(train_dataset.output_lang.n_words, experiment_best['HIDDEN_SIZE'],
                                    experiment_best['N_LAYERS'], experiment_best['RNN_TYPE'],
                                    experiment_best['DROPOUT'],
                                    experiment_best['ATTENTION']).to(device)

        encoder, decoder = pipeline.train(train_dataset, encoder, decoder, n_iter, print_every=100, learning_rate=0.001,
                                          device=device)
        results.append(pipeline.evaluate(test_dataset, encoder, decoder, max_length=MAX_LENGTH, verbose=False))

    avg_accuracy = sum(results) / len(results)
    print('Average accuracy for experiment best: {}'.format(avg_accuracy))
    if wandb_log:
        wandb.run.summary["Average accuracy for experiment best"] = avg_accuracy


def test_sequence_length():
    # Test how generalization works for different lengths
    splits = [24, 25, 26, 27, 28, 30, 32, 33, 36, 40, 48]

    results = defaultdict(list)

    for _ in range(n_runs):
        input_lang = scan_dataset.Lang()
        output_lang = scan_dataset.Lang()

        train_dataset = scan_dataset.ScanDataset(
            split=scan_dataset.ScanSplit.LENGTH_SPLIT,
            input_lang=input_lang,
            output_lang=output_lang,
            train=True
        )

        encoder = models.EncoderRNN(train_dataset.input_lang.n_words, overall_best['HIDDEN_SIZE'], device,
                                    overall_best['N_LAYERS'], overall_best['RNN_TYPE'], overall_best['DROPOUT']).to(
            device)
        decoder = models.DecoderRNN(train_dataset.output_lang.n_words, overall_best['HIDDEN_SIZE'],
                                    overall_best['N_LAYERS'], overall_best['RNN_TYPE'], overall_best['DROPOUT'],
                                    overall_best['ATTENTION']).to(device)

        encoder, decoder = pipeline.train(train_dataset, encoder, decoder, n_iter, print_every=100, learning_rate=0.001,
                                          device=device)

        # Evaluate on various lengths
        for split in splits:
            test_dataset = scan_dataset.ScanDataset(
                split=scan_dataset.ScanSplit.LENGTH_SPLIT,
                split_variation=split,
                input_lang=input_lang,
                output_lang=output_lang,
                train=False
            )

            results[split].append(
                pipeline.evaluate(test_dataset, encoder, decoder, max_length=MAX_LENGTH, verbose=False))

    # Average results
    mean_results = {}
    for split, result in results.items():
        mean_results[split] = sum(result) / len(result)

    # Find standard deviation
    std_results = {}
    for split, result in results.items():
        std_results[split] = np.std(result)

    # Plot bar chart
    plt.bar(list(results.keys()), list(mean_results.values()), align='center', yerr=list(std_results.values()),
            capsize=5)
    plt.xlabel('Ground-truth action sequence length')
    plt.ylabel('Accuracy on new commands (%)')
    plt.ylim((0., 1.))

    if wandb_log:
        wandb.log({"Sequence length": plt})

    plt.show()

    # Print results
    for split, result in results.items():
        print('Split: {}, Accuracy: {}'.format(split, sum(result) / len(result)))


def test_command_length():
    # Test how generalization works for different command lengths
    splits = [4, 6, 7, 8, 9]

    results = defaultdict(list)

    for _ in range(n_runs):
        input_lang = scan_dataset.Lang()
        output_lang = scan_dataset.Lang()

        train_dataset = scan_dataset.ScanDataset(
            split=scan_dataset.ScanSplit.LENGTH_SPLIT,
            input_lang=input_lang,
            output_lang=output_lang,
            train=True
        )

        encoder = models.EncoderRNN(train_dataset.input_lang.n_words, overall_best['HIDDEN_SIZE'], device,
                                    overall_best['N_LAYERS'], overall_best['RNN_TYPE'], overall_best['DROPOUT']).to(
            device)
        decoder = models.DecoderRNN(train_dataset.output_lang.n_words, overall_best['HIDDEN_SIZE'],
                                    overall_best['N_LAYERS'], overall_best['RNN_TYPE'], overall_best['DROPOUT'],
                                    attention=overall_best['ATTENTION'], device=device).to(device)
        encoder, decoder = pipeline.train(train_dataset, encoder, decoder, n_iter, print_every=100, learning_rate=0.001,
                                          device=device)

        # Evaluate on various lengths
        for split in splits:
            test_dataset = scan_dataset.ScanDataset(
                split=scan_dataset.ScanSplit.LENGTH_SPLIT,
                input_lang=input_lang,
                output_lang=output_lang,
                train=False
            )

            results[split].append(
                pipeline.evaluate(test_dataset, encoder, decoder, max_length=MAX_LENGTH, verbose=False))

    # Average results
    mean_results = {}
    for split, result in results.items():
        mean_results[split] = sum(result) / len(result)

    # Find standard deviation
    std_results = {}
    for split, result in results.items():
        std_results[split] = np.std(result)

    # Plot bar chart
    plt.bar(list(results.keys()), list(mean_results.values()), align='center', yerr=list(std_results.values()),
            capsize=5)
    plt.xlabel('Command length')
    plt.ylabel('Accuracy on new commands (%)')
    plt.ylim((0., 1.))

    if wandb_log:
        wandb.log({"Command length": plt})
    plt.show()

    # Print results
    for split, result in results.items():
        print('Split: {}, Accuracy: {}'.format(split, sum(result) / len(result)))


def main():
    # WANDB_API_KEY = os.environ.get('WANDB_API_KEY')
    if wandb_log:
        wandb.login()
        wandb.init(project="experiment-2", entity="atnlp")

    run_overall_best()
    run_experiment_best()
    # test_sequence_length()
    # test_command_length()


if __name__ == '__main__':
    main()
