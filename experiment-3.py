from torch.utils.data import DataLoader
import scan_dataset
import models
import pipeline
import torch

split_variations = list(reversed([1, 2, 4, 8, 16, 32]))
exp_3_variations = [1, 2, 3, 4, 5]

config = {
    'HIDDEN_SIZE': 100, # 25, 50, 100, 200, or 400
    'RNN_TYPE': 'LSTM', # RNN, GRU or LSTM
    'N_LAYERS': 1, # 1 or 2
    'DROPOUT': 0.1, # 0, 0.1 or 0.5
    'ATTENTION': True, # True or False
}

LEARNING_RATE = 1e-3
NUM_ITERATIONS = 10**5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for exp_3_variation in exp_3_variations:
    for split_variation in split_variations:
        input_lang = scan_dataset.Lang()
        output_lang = scan_dataset.Lang()

        
        train_dataset = scan_dataset.ScanDataset(
            split=scan_dataset.ScanSplit.ADD_PRIM_SPLIT,
            input_lang=input_lang,
            output_lang=output_lang,
            train=True,
            split_variation=split_variation,
            exp_3_variation=exp_3_variation
        )

        test_dataset = scan_dataset.ScanDataset(
            split=scan_dataset.ScanSplit.ADD_PRIM_SPLIT,
            input_lang=input_lang,
            output_lang=output_lang,
            train=False,
            split_variation=split_variation,
            exp_3_variation=exp_3_variation
        )
        
        MAX_LENGTH = max(train_dataset.input_lang.max_length, train_dataset.output_lang.max_length)
        
        encoder = models.EncoderRNN(
            train_dataset.input_lang.n_words, 
            config['HIDDEN_SIZE'], 
            MAX_LENGTH,
            device, 
            config['N_LAYERS'], 
            config['RNN_TYPE'], 
            config['DROPOUT']
        ).to(device)
        
        decoder = models.DecoderRNN(
            train_dataset.output_lang.n_words, 
            config['HIDDEN_SIZE'], 
            config['N_LAYERS'], 
            config['RNN_TYPE'], 
            config['DROPOUT'],
            config['ATTENTION']
        ).to(device)
        
        encoder, decoder = pipeline.train(train_dataset, encoder, decoder, NUM_ITERATIONS, learning_rate=LEARNING_RATE, device=device)
        eval_res = pipeline.evaluate(test_dataset, encoder, decoder, max_length=MAX_LENGTH, verbose=True)
        
        print(f"Split variation {split_variation}, Evaluation res: {eval_res}")
        print()
        