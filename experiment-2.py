from collections import defaultdict
import config
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
import Seq2SeqTrainer
import Seq2SeqModel
import Seq2SeqDataset



experiment_best = {
    'HIDDEN_SIZE': 50,  # 25, 50, 100, 200, or 400
    'RNN_TYPE': 'GRU',  # RNN, GRU or LSTM
    'N_LAYERS': 1,  # 1 or 2
    'DROPOUT': 0.5,  # 0, 0.1 or 0.5
    'ATTENTION': True,  # True or False
}

def length_generalization(splits, x_label='Ground-truth action sequence length', plot_title='Sequence length', oracle=False, experiment_best=False):
    results = defaultdict(list)

    for i in range(n_runs):
        encoder = pickle.load(open(f'runs/{"experiment" if experiment_best else "overall"}_best_encoder_exp_2_run_{i}.sav', 'rb'))
        decoder = pickle.load(open(f'runs/{"experiment" if experiment_best else "overall"}_best_decoder_exp_2_run_{i}.sav', 'rb'))

        # Evaluate on various lengths
        for split in splits:
            test_dataset = scan_dataset.ScanDataset(
                split=scan_dataset.ScanSplit.LENGTH_SPLIT,
                split_variation=split,
                input_lang=input_lang,
                output_lang=output_lang,
                train=False
            )
            if oracle:
                results[split].append(pipeline.oracle_eval(test_dataset, encoder, decoder, verbose=False, device=device))
            else:
                results[split].append(
                    pipeline.evaluate(test_dataset, encoder, decoder, max_length=MAX_LENGTH, verbose=False, device=device))
    
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


def test_sequence_length(oracle=False, experiment_best=False):
    # Test how generalization works for different lengths
    splits = [24, 25, 26, 27, 28, 30, 32, 33, 36, 40, 48]
    length_generalization(splits, oracle=oracle, experiment_best=experiment_best)


def test_command_length():
    # Test how generalization works for different command lengths
    splits = [4, 6, 7, 8, 9]
    length_generalization(splits, 'Command Length', 'Command Length')


def inspect_greedy_search(experiment_best=False):
    results = []

    for i in range(5): # n_runs
        encoder = pickle.load(open(f'runs/{"experiment" if experiment_best else "overall"}_best_encoder_exp_2_run_{i}.sav', 'rb'))
        decoder = pickle.load(open(f'runs/{"experiment" if experiment_best else "overall"}_best_decoder_exp_2_run_{i}.sav', 'rb'))


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

                encoder_outputs, encoder_hidden = encoder(input_tensor.to(device))

                decoder_input = torch.tensor([[scan_dataset.SOS_token]], device=device)

                decoder_hidden = encoder_hidden

                greedy_prob = 0

                MAX_LENGTH = 500
                for di in range(MAX_LENGTH):
                    decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden, encoder.all_hidden_states)

                    topv, topi = decoder_output.topk(1)
                    decode_log_prob = topv.squeeze().detach().item()
                    decoder_input = topi.detach()  # detach from history as input

                    greedy_prob += decode_log_prob

                    if decoder_input.item() == scan_dataset.EOS_token:
                        break
    
                decoder_hidden = encoder_hidden

                truth_prob = 0
                for di in range(target_length):
                    decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden, encoder.all_hidden_states)

                    prob = decoder_output.squeeze().detach()[target_tensor[di]].item()

                    truth_prob += prob
                    
                    decoder_input = target_tensor[di].unsqueeze(0)


                greedy_greatest = greedy_prob > truth_prob

                results[-1].append(greedy_greatest)

    # Calcate average amount of greedy search being greater than truth
    pct_greedy_greatest = [sum(data) / len(data) for data in results]
    avg_greedy_greatest = sum(pct_greedy_greatest) / len(pct_greedy_greatest)
    print(f'Average amount of greedy search being greater than truth for {"experiment" if experiment_best else "overall"} best: {avg_greedy_greatest}')


def oracle_test(experiment_best=False):

    results = []

    for i in range(5): # n_runs
        encoder = pickle.load(open(f'runs/{"experiment" if experiment_best else "overall"}_best_encoder_exp_2_run_{i}.sav', 'rb'))
        decoder = pickle.load(open(f'runs/{"experiment" if experiment_best else "overall"}_best_decoder_exp_2_run_{i}.sav', 'rb'))


        test_dataset = scan_dataset.ScanDataset(
            split=scan_dataset.ScanSplit.LENGTH_SPLIT,
            input_lang=input_lang,
            output_lang=output_lang,
            train=False
        )

        accuracy = pipeline.oracle_eval(test_dataset, encoder, decoder, verbose=False, device=device)

        results.append(accuracy)

    print(f'Oracle Accuracy for {"experiment" if experiment_best else "overall"} best: {np.mean(results)}')


def train_and_eval(
    model: Seq2SeqModel.Seq2SeqModel,
    train_args: Seq2SeqTrainer.Seq2SeqTrainingArguments):

    


        model = model()
        Seq2SeqModel.Seq2SeqModel()

        train_dataset = scan_dataset.ScanDataset(split = scan_dataset.ScanSplit.LENGTH_SPLIT, train = True)
        test_dataset = scan_dataset.ScanDataset(split = scan_dataset.ScanSplit.LENGTH_SPLIT, train = False)

        # Train and evaluate model
        trainer = Seq2SeqTrainer.Seq2SeqTrainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
        )
        
        model = trainer.train(evaluate_after=True)

        # Save model
        model.save(f'saved_models/experiment_2/run_{run}.sav', wandb_run=wandb_run, wandb_name=f'run_{run}')
    



def main():
    # WANDB_API_KEY = os.environ.get('WANDB_API_KEY')
    
    train_and_eval(
        
    )
    # run_overall_best()
    # run_experiment_best()
    # test_sequence_length()
    # test_command_length()

    # inspect_greedy_search(experiment_best=True)
    # inspect_greedy_search(experiment_best=False)
    # oracle_test()
    # oracle_test(experiment_best=True)
    # test_sequence_length(oracle=True, experiment_best=True)


if __name__ == '__main__':
    main()
