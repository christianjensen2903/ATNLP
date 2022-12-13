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
    for run in range(n_runs):
        encoder = models.EncoderRNN(train_dataset.input_lang.n_words, overall_best['HIDDEN_SIZE'], device,
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
    for _ in range(n_runs):
        encoder = models.EncoderRNN(train_dataset.input_lang.n_words, experiment_best['HIDDEN_SIZE'], device,
                                    experiment_best['N_LAYERS'], experiment_best['RNN_TYPE'],
                                    experiment_best['DROPOUT']).to(device)
        decoder = models.DecoderRNN(train_dataset.output_lang.n_words, experiment_best['HIDDEN_SIZE'],
                                    experiment_best['N_LAYERS'], experiment_best['RNN_TYPE'],
                                    experiment_best['DROPOUT'],
                                    experiment_best['ATTENTION']).to(device)

        encoder, decoder = pipeline.train(train_dataset, encoder, decoder, n_iter, print_every=100, learning_rate=0.001,
                                          device=device, log_wandb=True)
        pickle.dump(encoder, open('experiment_best_encoder_exp_2.sav', 'wb'))
        pickle.dump(decoder, open('experiment_best_decoder_exp_2.sav', 'wb'))
        results.append(pipeline.evaluate(test_dataset, encoder, decoder, max_length=MAX_LENGTH, verbose=False))

    avg_accuracy = sum(results) / len(results)
    print('Average accuracy for experiment best: {}'.format(avg_accuracy))
    if log_wandb:
        wandb.run.summary["Average accuracy for experiment best"] = avg_accuracy


def length_generalization(splits, x_label='Ground-truth action sequence length', plot_title='Sequence length'):
    results = defaultdict(list)

    for i in range(n_runs):
        encoder = pickle.load(open(f'runs/overall_best_encoder_exp_2_run_{i}.sav', 'rb'))
        decoder = pickle.load(open(f'runs/overall_best_decoder_exp_2_run_{i}.sav', 'rb'))

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
    
    print(f'{plot_title}: {results}')

    # Average results
    mean_results = {}
    for split, result in results.items():
        mean_results[split] = sum(result) / len(result)

    # Find standard deviation
    std_results = {}
    for split, result in results.items():
        std_results[split] = np.std(result)

    # Plot bar chart
    plt.bar(splits, list(mean_results.values()), align='center', yerr=list(std_results.values()),
            capsize=5)
    plt.xlabel(x_label)
    # TODO: figure out how to set x axis labels to exactly 'splits'
    # plt.xticks()
    plt.ylabel('Accuracy on new commands (%)')
    plt.ylim((0., 1.))

    if log_wandb:
        wandb.log({plot_title: plt})

    # plt.show()

    # Print results
    for split, result in results.items():
        print('Split: {}, Accuracy: {}'.format(split, sum(result) / len(result)))


def test_sequence_length():
    # Test how generalization works for different lengths
    splits = [24, 25, 26, 27, 28, 30, 32, 33, 36, 40, 48]
    length_generalization(splits)


def test_command_length():
    # Test how generalization works for different command lengths
    splits = [4, 6, 7, 8, 9]
    length_generalization(splits, 'Command Length', 'Command Length')


def inspect_greedy_search():
    results = []

    for i in range(5): # n_runs
        encoder = pickle.load(open(f'runs/overall_best_encoder_exp_2_run_{i}.sav', 'rb'))
        decoder = pickle.load(open(f'runs/overall_best_decoder_exp_2_run_{i}.sav', 'rb'))


        test_dataset = scan_dataset.ScanDataset(
            split=scan_dataset.ScanSplit.LENGTH_SPLIT,
            input_lang=input_lang,
            output_lang=output_lang,
            train=False
        )

        encoder.eval()
        decoder.eval()

        results.append([])

        with torch.no_grad():
            for input_tensor, target_tensor in tqdm(test_dataset, total=len(test_dataset), leave=False, desc="Inspecting"):
                input_tensor, target_tensor = test_dataset.convert_to_tensor(input_tensor, target_tensor)

                target_length = target_tensor.size(0)

                encoder_hidden, encoder_hidden_all = encoder(input_tensor.to(device))

                decoder_input = torch.tensor([[scan_dataset.SOS_token]], device=device)

                decoder_hidden = encoder_hidden

                greedy_prob = 1.0

                MAX_LENGTH = 100
                for di in range(MAX_LENGTH):
                    decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden, encoder_hidden_all)

                    topv, topi = decoder_output.topk(1)
                    decode_log_prob = topv.squeeze().detach().item()
                    decoder_input = topi.squeeze().detach()  # detach from history as input

                    greedy_prob *= decode_log_prob

                    if decoder_input.item() == scan_dataset.EOS_token:
                        break


                truth_prob = 1.0
                for di in range(target_length):
                    decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden, encoder_hidden_all)

                    prob = decoder_output.squeeze().detach()[target_tensor[di]].item()

                    truth_prob *= prob

                    decoder_input = target_tensor[di]
                    

                greedy_greatest = greedy_prob > truth_prob

                results[-1].append(greedy_greatest)

    # Calcate average amount of greedy search being greater than truth
    pct_greedy_greatest = [sum(data) / len(data) for data in results]
    avg_greedy_greatest = sum(pct_greedy_greatest) / len(pct_greedy_greatest)
    print('Average amount of greedy search being greater than truth: {}'.format(avg_greedy_greatest))


def main():
    # WANDB_API_KEY = os.environ.get('WANDB_API_KEY')
    if log_wandb:
        wandb.login()
        wandb.init(project="experiment-2", entity="atnlp")

    # run_overall_best()
    # run_experiment_best()
    # test_sequence_length()
    # test_command_length()

    inspect_greedy_search()


if __name__ == '__main__':
    main()
