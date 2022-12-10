import scan_dataset
import models
import pipeline
import torch
from matplotlib import pyplot as plt
# import wandb
import os

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

n_iter = 100


overall_best = {
    'HIDDEN_SIZE': 200, # 25, 50, 100, 200, or 400
    'RNN_TYPE': 'LSTM', # RNN, GRU or LSTM
    'N_LAYERS': 2, # 1 or 2
    'DROPOUT': 0.5, # 0, 0.1 or 0.5
    'ATTENTION': False, # True or False
}

experiment_best = {
    'HIDDEN_SIZE': 200,
    'RNN_TYPE': 'LSTM',
    'N_LAYERS': 2,
    'DROPOUT': 0,
    'ATTENTION': False,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')


# WANDB_API_KEY = os.environ.get('WANDB_API_KEY')
# wandb.login()

# wandb.init(project="experiment-1", entity="atnlp")

results = []
# Train 5 times and average the results
for _ in range(5):
    encoder = models.EncoderRNN(train_dataset.input_lang.n_words, overall_best['HIDDEN_SIZE'], device, overall_best['N_LAYERS'], overall_best['RNN_TYPE'], overall_best['DROPOUT']).to(device)
    decoder = models.DecoderRNN(train_dataset.output_lang.n_words, overall_best['HIDDEN_SIZE'], overall_best['N_LAYERS'], overall_best['RNN_TYPE'], overall_best['DROPOUT'],overall_best['ATTENTION']).to(device)

    encoder, decoder = pipeline.train(train_dataset, encoder, decoder, n_iter, print_every=100, learning_rate=0.001, device=device)
    results.append(pipeline.evaluate(test_dataset, encoder, decoder, max_length=MAX_LENGTH, verbose=False, device=device))

avg_accuracy = sum(results) / len(results)
print('Average accuracy for overall best: {}'.format(avg_accuracy))
# wandb.run.summary["Average accuracy for overall best"] = avg_accuracy


results = []
# Train 5 times and average the results
for _ in range(5):
    encoder = models.EncoderRNN(train_dataset.input_lang.n_words, experiment_best['HIDDEN_SIZE'], device, experiment_best['N_LAYERS'], experiment_best['RNN_TYPE'], experiment_best['DROPOUT']).to(device)
    decoder = models.DecoderRNN(train_dataset.output_lang.n_words, experiment_best['HIDDEN_SIZE'], experiment_best['N_LAYERS'], experiment_best['RNN_TYPE'], experiment_best['DROPOUT'],experiment_best['ATTENTION']).to(device)

    encoder, decoder = pipeline.train(train_dataset, encoder, decoder, n_iter, print_every=100, learning_rate=0.001, device=device)
    results.append(pipeline.evaluate(test_dataset, encoder, decoder, max_length=MAX_LENGTH, verbose=False))

avg_accuracy = sum(results) / len(results)
print('Average accuracy for experiment best: {}'.format(avg_accuracy))
wandb.run.summary["Average accuracy for experiment best"] = avg_accuracy




splits = ['p1', 'p2', 'p4', 'p8', 'p16', 'p32', 'p64']


results = {}

for split in splits:
    results[split] = []
    for _ in range(5):
        input_lang = scan_dataset.Lang()
        output_lang = scan_dataset.Lang()

        train_dataset = scan_dataset.ScanDataset(
            split=scan_dataset.ScanSplit.SIMPLE_SPLIT,
            split_variation=split,
            input_lang=input_lang,
            output_lang=output_lang,
            train=True
        )

        test_dataset = scan_dataset.ScanDataset(
            split=scan_dataset.ScanSplit.SIMPLE_SPLIT,
            split_variation=split,
            input_lang=input_lang,
            output_lang=output_lang,
            train=False
        )

        encoder = models.EncoderRNN(train_dataset.input_lang.n_words, overall_best['HIDDEN_SIZE'], device, overall_best['N_LAYERS'], overall_best['RNN_TYPE'], overall_best['DROPOUT']).to(device)
        decoder = models.DecoderRNN(train_dataset.output_lang.n_words, overall_best['HIDDEN_SIZE'], overall_best['N_LAYERS'], overall_best['RNN_TYPE'], experiment_best['DROPOUT'],experiment_best['ATTENTION']).to(device)

        encoder, decoder = pipeline.train(train_dataset, encoder, decoder, n_iter, print_every=100, learning_rate=0.001, device=device)
        results[split].append(pipeline.evaluate(test_dataset, encoder, decoder, max_length=100, verbose=False))


# Plot bar chart
plt.bar(range(len(results)), list(results.values()), align='center')
plt.xticks(range(len(results)), list(results.keys()))
plt.show()

# Print results
for split, result in results.items():
    print('Split: {}, Accuracy: {}'.format(split, sum(result) / len(result)))
