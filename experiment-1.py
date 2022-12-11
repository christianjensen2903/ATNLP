import scan_dataset
import models
import pipeline
import torch
#import wandb
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

#wandb.login()

#wandb.init(project="experiment-1", entity="atnlp")

def get_datasets(split_variation):
    input_lang = scan_dataset.Lang()
    output_lang = scan_dataset.Lang()

    train_dataset = scan_dataset.ScanDataset(
        split=scan_dataset.ScanSplit.SIMPLE_SPLIT,
        input_lang=input_lang,
        output_lang=output_lang,
        train=True,
        split_variation=split_variation
    )

    test_dataset = scan_dataset.ScanDataset(
        split=scan_dataset.ScanSplit.SIMPLE_SPLIT,
        input_lang=input_lang,
        output_lang=output_lang,
        train=False,
        split_variation=split_variation
    )
    
    return train_dataset, test_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

experiment_1_config = dict(HIDDEN_SIZE=200, N_LAYERS=2, DROPOUT=0, RNN_TYPE='LSTM', ATTENTION=False)
overall_best_config = dict(HIDDEN_SIZE=200, N_LAYERS=2, DROPOUT=0.5, RNN_TYPE='LSTM', ATTENTION=False)

SPLIT_VARIATIONS = ['p1', 'p2', 'p4', 'p8', 'p16', 'p32', 'p64']
NUM_EXPERIMENTS = 5

LEARNING_RATE = 1e-3
NUM_ITERATIONS = 10**5
NUM_EXPERIMENTS = 5

# ============== Train the overall best five times and average the results ==============
results = []
for _ in range(NUM_EXPERIMENTS):
    train_dataset, test_dataset = get_datasets(None)
    MAX_LENGTH = max(train_dataset.input_lang.max_length, train_dataset.output_lang.max_length)
    
    encoder = models.EncoderRNN(train_dataset.input_lang.n_words, config=overall_best_config).to(device)
        
    decoder = models.DecoderRNN(train_dataset.output_lang.n_words, config=overall_best_config).to(device)

    encoder, decoder = pipeline.train(train_dataset, encoder, decoder, NUM_ITERATIONS, verbose=False, learning_rate=LEARNING_RATE, device=device)
    eval_res = pipeline.evaluate(test_dataset, encoder, decoder, max_length=MAX_LENGTH, verbose=False, device=device)
    print("Eval res", eval_res)

    results.append(eval_res)
    
avg_accuracy = sum(results) / len(results)
print('Average accuracy for overall best: {}'.format(avg_accuracy))

#wandb.run.summary["Average accuracy for overall best"] = avg_accuracy

# ============== Train the best of the experiment five times and average the results ==============
results = []
for _ in range(NUM_EXPERIMENTS):
    train_dataset, test_dataset = get_datasets(None)
    MAX_LENGTH = max(train_dataset.input_lang.max_length, train_dataset.output_lang.max_length)
    
    encoder = models.EncoderRNN(
            train_dataset.input_lang.n_words, 
            experiment_1_config['HIDDEN_SIZE'], 
            device, 
            experiment_1_config['N_LAYERS'],
            experiment_1_config['RNN_TYPE'], 
            experiment_1_config['DROPOUT']
        ).to(device)
        
    decoder = models.DecoderRNN(
        train_dataset.output_lang.n_words, 
        experiment_1_config['HIDDEN_SIZE'], 
        experiment_1_config['N_LAYERS'], 
        experiment_1_config['RNN_TYPE'], 
        experiment_1_config['DROPOUT'],
        experiment_1_config['ATTENTION']
    ).to(device)

    encoder, decoder = pipeline.train(train_dataset, encoder, decoder, NUM_ITERATIONS, verbose=False, learning_rate=LEARNING_RATE, device=device)
    eval_res = pipeline.evaluate(test_dataset, encoder, decoder, max_length=MAX_LENGTH, verbose=False, device=device)

    results.append(eval_res)
    
avg_accuracy = sum(results) / len(results)
print('Average accuracy for overall best: {}'.format(avg_accuracy))

#wandb.run.summary["Average accuracy for experiment best"] = avg_accuracy

# ============== Running each split variation five times for the overall best model and average the results ==============
results = {split_variation : [] for split_variation in SPLIT_VARIATIONS}
for _ in range(NUM_EXPERIMENTS):
    for split_variation in tqdm(SPLIT_VARIATIONS, desc="Training ratio", leave=False):
        train_dataset, test_dataset = get_datasets(split_variation)
        MAX_LENGTH = max(train_dataset.input_lang.max_length, train_dataset.output_lang.max_length)
        
        encoder = models.EncoderRNN(
            train_dataset.input_lang.n_words, 
            overall_best_config['HIDDEN_SIZE'], 
            device, 
            overall_best_config['N_LAYERS'],
            overall_best_config['RNN_TYPE'], 
            overall_best_config['DROPOUT']
        ).to(device)
        
        decoder = models.DecoderRNN(
            train_dataset.output_lang.n_words, 
            overall_best_config['HIDDEN_SIZE'], 
            overall_best_config['N_LAYERS'], 
            overall_best_config['RNN_TYPE'], 
            overall_best_config['DROPOUT'],
            overall_best_config['ATTENTION']
        ).to(device)

        encoder, decoder = pipeline.train(train_dataset, encoder, decoder, NUM_ITERATIONS, verbose=False, learning_rate=LEARNING_RATE, device=device)
        eval_res = pipeline.evaluate(test_dataset, encoder, decoder, max_length=MAX_LENGTH, verbose=False, device=device)

        results[split_variation].append(eval_res)
        
# Average results
mean_results = {}
for split, result in results.items():
    mean_results[split] = sum(result) / len(result)

# Find standard deviation
std_results = {}
for split, result in results.items():
    std_results[split] = np.std(result)

# Plot bar chart
plt.bar(list(results.keys()), list(mean_results.values()), align='center', yerr=list(std_results.values()), capsize=5)
plt.xlabel('Percent of commands used for training')
plt.ylabel('Accuracy on new commands (%)')

#wandb.log({"Percent commands": plt})
plt.show()

# Print results
for split, result in results.items():
    print('Split: {}, Accuracy: {}'.format(split, sum(result) / len(result)))