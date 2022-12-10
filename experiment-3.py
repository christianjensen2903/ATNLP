from torch.utils.data import DataLoader
import scan_dataset
import models
import pipeline
import torch

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


overall_best = {
    'HIDDEN_SIZE': 200, # 25, 50, 100, 200, or 400
    'RNN_TYPE': 'LSTM', # RNN, GRU or LSTM
    'N_LAYERS': 2, # 1 or 2
    'DROPOUT': 0.5, # 0, 0.1 or 0.5
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = models.EncoderRNN(train_dataset.input_lang.n_words, overall_best['HIDDEN_SIZE'], device, overall_best['N_LAYERS'], overall_best['RNN_TYPE'], overall_best['DROPOUT']).to(device)
decoder = models.DecoderRNN(train_dataset.output_lang.n_words, overall_best['HIDDEN_SIZE'], overall_best['N_LAYERS'], overall_best['RNN_TYPE'], overall_best['DROPOUT']).to(device)

encoder, decoder = pipeline.train(train_dataset, encoder, decoder, 10000, print_every=100, learning_rate=0.001, device=device)

pipeline.evaluate(test_dataset, encoder, decoder, max_length=100, verbose=True)

