from collections import defaultdict
import config
import scan_dataset
import RNNSeq2Seq
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
import helper
from typing import List, Dict, Tuple, Union, Optional
from CustomTrainerCallback import CustomTrainerCallback
import config

n_runs = 1

def length_generalization(
    model: Seq2SeqModel.Seq2SeqModel,
    train_args: Seq2SeqTrainer.Seq2SeqTrainingArguments,
    run_type: str,
    splits: List[int],
    input_lang: scan_dataset.Lang,
    output_lang: scan_dataset.Lang,):

    results: Dict[int, List] = {}

    for i in range(n_runs):

        custom_callback = CustomTrainerCallback(run_index=i, run_type=run_type)

        # Load model
        model = model.load(path=f'saved-models/{custom_callback._run_to_string()}')

        # Evaluate on various lengths
        for split in splits:
            test_dataset = scan_dataset.ScanDataset(
                split=scan_dataset.ScanSplit.LENGTH_SPLIT,
                split_variation=split,
                input_lang=input_lang,
                output_lang=output_lang,
                train=False
            )

            trainer = Seq2SeqTrainer.Seq2SeqTrainer(model=model,args=train_args,test_dataset=test_dataset,)
            metrics = trainer.evaluate()
            if split not in results:
                results[split] = [metrics['eval_accuracy']]
            else:
                results[split].append(metrics['eval_accuracy'])

    # Plot results
    helper.plot_bar_chart(
        results,
        plot_title=f'Accuracy on {run_type} Split',
        x_label=run_type,
        y_label='Accuracy',

    )

    # Print results
    for split, result in results.items():
        print('Split: {}, Accuracy: {}'.format(split, sum(result) / len(result)))


def test_sequence_length(model: Seq2SeqModel.Seq2SeqModel,
    train_args: Seq2SeqTrainer.Seq2SeqTrainingArguments,
    run_type: str,
    input_lang: scan_dataset.Lang,
    output_lang: scan_dataset.Lang,):
    # Test how generalization works for different lengths
    splits = [24, 25, 26, 27, 28, 30, 32, 33, 36, 40, 48]
    length_generalization(
        model=model,
        train_args=train_args,
        run_type=run_type,
        splits=splits,
        input_lang=input_lang,
        output_lang=output_lang,
    )


# def inspect_greedy_search(experiment_best=False):
#     results = []

#     for i in range(5): # n_runs
#         encoder = pickle.load(open(f'runs/{"experiment" if experiment_best else "overall"}_best_encoder_exp_2_run_{i}.sav', 'rb'))
#         decoder = pickle.load(open(f'runs/{"experiment" if experiment_best else "overall"}_best_decoder_exp_2_run_{i}.sav', 'rb'))


#         test_dataset = scan_dataset.ScanDataset(
#             split=scan_dataset.ScanSplit.LENGTH_SPLIT,
#             input_lang=input_lang,
#             output_lang=output_lang,
#             train=False
#         )

#         encoder.eval()
#         decoder.eval()

#         results.append([])

#         with torch.no_grad():
#             for input_tensor, target_tensor in tqdm(test_dataset, total=len(test_dataset), leave=False, desc="Inspecting"):
#                 input_tensor, target_tensor = test_dataset.convert_to_tensor(input_tensor, target_tensor)

#                 target_length = target_tensor.size(0)

#                 encoder_outputs, encoder_hidden = encoder(input_tensor.to(device))

#                 decoder_input = torch.tensor([[scan_dataset.SOS_token]], device=device)

#                 decoder_hidden = encoder_hidden

#                 greedy_prob = 0

#                 MAX_LENGTH = 500
#                 for di in range(MAX_LENGTH):
#                     decoder_output, decoder_hidden = decoder(
#                         decoder_input, decoder_hidden, encoder.all_hidden_states)

#                     topv, topi = decoder_output.topk(1)
#                     decode_log_prob = topv.squeeze().detach().item()
#                     decoder_input = topi.detach()  # detach from history as input

#                     greedy_prob += decode_log_prob

#                     if decoder_input.item() == scan_dataset.EOS_token:
#                         break
    
#                 decoder_hidden = encoder_hidden

#                 truth_prob = 0
#                 for di in range(target_length):
#                     decoder_output, decoder_hidden = decoder(
#                         decoder_input, decoder_hidden, encoder.all_hidden_states)

#                     prob = decoder_output.squeeze().detach()[target_tensor[di]].item()

#                     truth_prob += prob
                    
#                     decoder_input = target_tensor[di].unsqueeze(0)


#                 greedy_greatest = greedy_prob > truth_prob

#                 results[-1].append(greedy_greatest)

#     # Calcate average amount of greedy search being greater than truth
#     pct_greedy_greatest = [sum(data) / len(data) for data in results]
#     avg_greedy_greatest = sum(pct_greedy_greatest) / len(pct_greedy_greatest)
#     print(f'Average amount of greedy search being greater than truth for {"experiment" if experiment_best else "overall"} best: {avg_greedy_greatest}')


# def oracle_test(experiment_best=False):

#     results = []

#     for i in range(5): # n_runs
#         encoder = pickle.load(open(f'runs/{"experiment" if experiment_best else "overall"}_best_encoder_exp_2_run_{i}.sav', 'rb'))
#         decoder = pickle.load(open(f'runs/{"experiment" if experiment_best else "overall"}_best_decoder_exp_2_run_{i}.sav', 'rb'))


#         test_dataset = scan_dataset.ScanDataset(
#             split=scan_dataset.ScanSplit.LENGTH_SPLIT,
#             input_lang=input_lang,
#             output_lang=output_lang,
#             train=False
#         )

#         accuracy = pipeline.oracle_eval(test_dataset, encoder, decoder, verbose=False, device=device)

#         results.append(accuracy)

#     print(f'Oracle Accuracy for {"experiment" if experiment_best else "overall"} best: {np.mean(results)}')


def experiment_loop(
    model: Seq2SeqModel.Seq2SeqModel,
    model_config: Seq2SeqModel.Seq2SeqModelConfig,
    train_args: Seq2SeqTrainer.Seq2SeqTrainingArguments,
    run_type: str,
    input_lang: scan_dataset.Lang,
    output_lang: scan_dataset.Lang,):

    accuracies = []

    for i in range(n_runs):
        # Load dataset
        train_dataset = scan_dataset.ScanDataset(input_lang = input_lang, output_lang = output_lang, split = scan_dataset.ScanSplit.LENGTH_SPLIT, train = True)
        test_dataset = scan_dataset.ScanDataset(input_lang = input_lang, output_lang = output_lang, split = scan_dataset.ScanSplit.LENGTH_SPLIT, train = False)

        model_config.input_vocab_size = input_lang.n_words
        model_config.output_vocab_size = output_lang.n_words
        model_config.pad_index = input_lang.pad_index
        model_config.sos_index = input_lang.sos_index
        model_config.eos_index = input_lang.eos_index

        # Reset model
        model.from_config(config=model_config)

        custom_callback = CustomTrainerCallback(run_index=i, run_type=run_type)
        
        # Train and evaluate model
        trainer = Seq2SeqTrainer.Seq2SeqTrainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            callbacks=[custom_callback],
        )
        
        model = trainer.train(evaluate_after=False)
        model.save(path=f'saved-models/{custom_callback._run_to_string()}/')
        metrics = trainer.evaluate()
        accuracies.append(metrics['eval_accuracy'])

    print(f'Average accuracy for {run_type}: {np.mean(accuracies)}')



def main():

    train_args = config.paper_train_args

    if train_args.log_wandb:
        wandb.login()
        wandb.init(project="experiment-2", entity="atnlp", config=train_args)
    
    input_lang, output_lang = Seq2SeqDataset.Lang(), Seq2SeqDataset.Lang()
    # experiment_loop(
    #     model=RNNSeq2Seq.RNNSeq2Seq(),
    #     model_config=config.overall_best_config,
    #     train_args=train_args,
    #     run_type='overall_best',
    #     input_lang=input_lang,
    #     output_lang=output_lang,
    # )

    test_sequence_length(
        model=RNNSeq2Seq.RNNSeq2Seq(),
        train_args=train_args,
        input_lang=input_lang,
        output_lang=output_lang,
        run_type='overall_best',
    )

    experiment_best_config = RNNSeq2Seq.RNNSeq2SeqConfig(
        hidden_size=50,
        rnn_type='GRU',
        n_layers=1,
        dropout_p=0.5,
        attention=True,
    )

    experiment_loop(
        model=RNNSeq2Seq.RNNSeq2Seq(),
        model_config=experiment_best_config,
        train_args=train_args,
        run_type='experiment_best',
        input_lang=input_lang,
        output_lang=output_lang,
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
