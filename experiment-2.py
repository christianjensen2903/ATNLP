from torch.utils.data import DataLoader
import scan_dataset
import models
import pipeline
import torch
import wandb
import os
from matplotlib import pyplot as plt

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


overall_best = {
    'HIDDEN_SIZE': 200, # 25, 50, 100, 200, or 400
    'RNN_TYPE': 'LSTM', # RNN, GRU or LSTM
    'N_LAYERS': 2, # 1 or 2
    'DROPOUT': 0.5, # 0, 0.1 or 0.5
    'ATTENTION': False, # True or False
}

experiment_best = {
    'HIDDEN_SIZE': 50, # 25, 50, 100, 200, or 400
    'RNN_TYPE': 'GRU', # RNN, GRU or LSTM
    'N_LAYERS': 1, # 1 or 2
    'DROPOUT': 0.5, # 0, 0.1 or 0.5
    'ATTENTION': True, # True or False
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# WANDB_API_KEY = os.environ.get('WANDB_API_KEY')

wandb.init(project="experiment-2", entity="atnlp")

results = []
# Train 5 times and average the results
for _ in range(5):
    encoder = models.EncoderRNN(train_dataset.input_lang.n_words, overall_best['HIDDEN_SIZE'], device, overall_best['N_LAYERS'], overall_best['RNN_TYPE'], overall_best['DROPOUT']).to(device)
    decoder = models.DecoderRNN(train_dataset.output_lang.n_words, overall_best['HIDDEN_SIZE'], overall_best['N_LAYERS'], overall_best['RNN_TYPE'], overall_best['DROPOUT'],overall_best['ATTENTION']).to(device)

    encoder, decoder = pipeline.train(train_dataset, encoder, decoder, 1000, print_every=100, learning_rate=0.001, device=device)
    results.append(pipeline.evaluate(test_dataset, encoder, decoder, max_length=MAX_LENGTH, verbose=False))

avg_accuracy = sum(results) / len(results)
print('Average accuracy for overall best: {}'.format(avg_accuracy))
wandb.run.summary["Average accuracy for overall best"] = avg_accuracy


results = []
# Train 5 times and average the results
for _ in range(5):
    encoder = models.EncoderRNN(train_dataset.input_lang.n_words, experiment_best['HIDDEN_SIZE'], device, experiment_best['N_LAYERS'], experiment_best['RNN_TYPE'], experiment_best['DROPOUT']).to(device)
    decoder = models.DecoderRNN(train_dataset.output_lang.n_words, experiment_best['HIDDEN_SIZE'], experiment_best['N_LAYERS'], experiment_best['RNN_TYPE'], experiment_best['DROPOUT'],experiment_best['ATTENTION']).to(device)

    encoder, decoder = pipeline.train(train_dataset, encoder, decoder, 1000, print_every=100, learning_rate=0.001, device=device)
    results.append(pipeline.evaluate(test_dataset, encoder, decoder, max_length=MAX_LENGTH, verbose=False))

avg_accuracy = sum(results) / len(results)
print('Average accuracy for experiment best: {}'.format(avg_accuracy))
wandb.run.summary["Average accuracy for experiment best"] = avg_accuracy




# Test how generalization works for different lengths

splits = [24, 25, 26, 27, 28, 29, 30, 32, 33, 36, 40, 48]

results = {}

for _ in range(5):
    input_lang = scan_dataset.Lang()
    output_lang = scan_dataset.Lang()

    train_dataset = scan_dataset.ScanDataset(
        split=scan_dataset.ScanSplit.SIMPLE_SPLIT,
        input_lang=input_lang,
        output_lang=output_lang,
        train=True
    )

    encoder = models.EncoderRNN(train_dataset.input_lang.n_words, overall_best['HIDDEN_SIZE'], device, overall_best['N_LAYERS'], overall_best['RNN_TYPE'], overall_best['DROPOUT']).to(device)
    decoder = models.DecoderRNN(train_dataset.output_lang.n_words, overall_best['HIDDEN_SIZE'], overall_best['N_LAYERS'], overall_best['RNN_TYPE'], experiment_best['DROPOUT'],experiment_best['ATTENTION']).to(device)
    encoder, decoder = pipeline.train(train_dataset, encoder, decoder, 1000, print_every=100, learning_rate=0.001, device=device)

    # Evaluate on various lengths
    for split in splits:
        results[split] = []
        
        test_dataset = scan_dataset.ScanDataset(
            split=scan_dataset.ScanSplit.SIMPLE_SPLIT,
            split_variation=split,
            input_lang=input_lang,
            output_lang=output_lang,
            train=False
        )

        results[split].append(pipeline.evaluate(test_dataset, encoder, decoder, max_length=MAX_LENGTH, verbose=False))


# Plot bar chart
plt.bar(range(len(results)), list(results.values()), align='center')
plt.xticks(range(len(results)), list(results.keys()))
plt.show()

# Print results
for split, result in results.items():
    print('Split: {}, Accuracy: {}'.format(split, sum(result) / len(result)))




# Test how generalization works for different command lengths

splits = [4, 6, 7, 8, 9]

results = {}

for _ in range(5):
    input_lang = scan_dataset.Lang()
    output_lang = scan_dataset.Lang()

    train_dataset = scan_dataset.ScanDataset(
        split=scan_dataset.ScanSplit.SIMPLE_SPLIT,
        input_lang=input_lang,
        output_lang=output_lang,
        train=True
    )

    encoder = models.EncoderRNN(train_dataset.input_lang.n_words, overall_best['HIDDEN_SIZE'], device, overall_best['N_LAYERS'], overall_best['RNN_TYPE'], overall_best['DROPOUT']).to(device)
    decoder = models.DecoderRNN(train_dataset.output_lang.n_words, overall_best['HIDDEN_SIZE'], overall_best['N_LAYERS'], overall_best['RNN_TYPE'], experiment_best['DROPOUT'],experiment_best['ATTENTION']).to(device)
    encoder, decoder = pipeline.train(train_dataset, encoder, decoder, 1000, print_every=100, learning_rate=0.001, device=device)

    # Evaluate on various lengths
    for split in splits:
        results[split] = []
        
        test_dataset = scan_dataset.ScanDataset(
            split=scan_dataset.ScanSplit.SIMPLE_SPLIT,
            input_lang=input_lang,
            output_lang=output_lang,
            train=False
        )

        # Filter out sequences with different command lengths
        new_X, new_y = [], []
        for i in range(len(test_dataset)):
            X, y = test_dataset.convert_to_tensor(test_dataset[i])
            if len((X)) == split:
                new_X.append(test_dataset[i][0])
                new_y.append(test_dataset[i][1])

        test_dataset.X = new_X
        test_dataset.y = new_y


        results[split].append(pipeline.evaluate(test_dataset, encoder, decoder, max_length=MAX_LENGTH, verbose=False))


# Plot bar chart
plt.bar(range(len(results)), list(results.values()), align='center')
plt.xticks(range(len(results)), list(results.keys()))
plt.show()

# Print results
for split, result in results.items():
    print('Split: {}, Accuracy: {}'.format(split, sum(result) / len(result)))