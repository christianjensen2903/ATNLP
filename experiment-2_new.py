import config
import scan_dataset
import RNNSeq2Seq
import torch
import wandb
import numpy as np
from tqdm import tqdm
import Seq2SeqTrainer
import Seq2SeqModel
from typing import List, Dict, Tuple, Union, Optional
from CustomTrainerCallback import CustomTrainerCallback
import config
from ExperimentBase import ExperimentBase


class Experiment2(ExperimentBase):

    def __init__(self,
                 model: Seq2SeqModel.Seq2SeqModel,
                 model_config: Seq2SeqModel.Seq2SeqModelConfig,
                 train_args: Seq2SeqTrainer.Seq2SeqTrainingArguments,
                 run_type: str,
                 n_runs: int,
                  ):
        self.split = scan_dataset.ScanSplit.LENGTH_SPLIT
        super().__init__(model, model_config, train_args, run_type, n_runs, split=self.split)

    def run(self):
        self.train_models()
        self.length_generalization(split=self.split, splits=[24, 25, 26, 27, 28, 30, 32, 33, 36, 40, 48], x_label='Ground-truth action sequence length', plot_title=f'sequence_split')
        self.length_generalization(split=self.split, splits=[4, 6, 7, 8, 9], x_label='Command length', plot_title=f'command_split')
        self.inspect_greedy_search()
        self.oracle_test()


    def inspect_greedy_search(self):
        results = []

        for i, model in enumerate(self.saved_models):

            model.eval()

            oracle_best = []  # number of times oracle is better than greedy search
            with torch.no_grad():
                for input, target in tqdm(self.test_dataset, total=len(self.test_dataset), leave=False, desc="Evaluating"):
                    input_tensor, target_tensor = self.test_dataset.convert_to_tensor(input, target)

                    _, greedy_prob = model.predict(input_tensor, max_length=100)

                    _, oracle_prob = model.predict(input_tensor, oracle_target=target_tensor)

                    oracle_best.append(oracle_prob > greedy_prob)

            results.append(oracle_best)


        # Calcate average amount of oracle search being greater than truth
        pct_oracle_best = [sum(data) / len(data) for data in results]
        avg_oracle_best = sum(pct_oracle_best) / len(pct_oracle_best)
        if self.train_args.log_wandb:
            wandb.log({'oracle_search_pct': avg_oracle_best})
        print(f'Average amount of oracle search being greater than greedy: {avg_oracle_best}')



    def oracle_test(self):
        results = []

        for i, model in enumerate(self.saved_models):

            model.eval()

            n_correct = []  # number of correct predictions
            with torch.no_grad():
                for input, target in tqdm(self.test_dataset, total=len(self.test_dataset), leave=False, desc="Evaluating with oracle"):
                    input_tensor, target_tensor = self.test_dataset.convert_to_tensor(input, target)

                    pred, _  = model.predict(input_tensor, oracle_length=target_tensor.size(1))

                    pred = pred.squeeze().cpu().numpy()
                    ground_truth = target_tensor.numpy().squeeze()
                    
                    n_correct.append(np.all(pred == ground_truth))

            accuracy = np.mean(n_correct)
            results.append(accuracy)

        
        avg_accuracy = np.mean(results)

        if self.train_args.log_wandb:
            wandb.log({'oracle_accuracy': avg_accuracy})

        print(f'Average accuracy with oracle: {avg_accuracy}')

            


def main():

    train_args = config.paper_train_args

    # Initialize wandb

    if train_args.log_wandb:
        wandb.init(
            project="experiment-2",
            entity="atnlp",
            config=train_args,
            reinit=True,
            tags=['experiment-2', 'overall_best'])

    Experiment2(
        model=RNNSeq2Seq.RNNSeq2Seq(),
        model_config=config.overall_best_config,
        train_args=train_args,
        run_type='overall_best',
        n_runs=5,
    ).run()




if __name__ == '__main__':
    main()
