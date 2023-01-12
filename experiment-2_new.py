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
from ExperimentBase import ExperimentBase


class Experiment2(ExperimentBase):

    def __init__(self,
                 model: Seq2SeqModel.Seq2SeqModel,
                 model_config: Seq2SeqModel.Seq2SeqModelConfig,
                 train_args: Seq2SeqTrainer.Seq2SeqTrainingArguments,
                 run_type: str,
                 n_runs: int,
                  ):
        super().__init__(model, model_config, train_args, run_type, n_runs, split=scan_dataset.ScanSplit.LENGTH_SPLIT)

    def run(self):
        self.train_models()
        # self.length_generalization(splits=[24, 25, 26, 27, 28, 30, 32, 33, 36, 40, 48], x_label='Ground-truth action sequence length', plot_title=f'Accuracy on {self.run_type} Split')
        # self.length_generalization(splits=[4, 6, 7, 8, 9], x_label='Command length', plot_title=f'Accuracy on {self.run_type} Split')
        self.inspect_greedy_search()

    def length_generalization(self, splits: List[int], x_label: str = '', plot_title: str = ''):

        results: Dict[int, List] = {}

        for i, model in enumerate(self.saved_models):

            # Evaluate on various lengths
            for split in splits:
                test_dataset = scan_dataset.ScanDataset(
                    split=scan_dataset.ScanSplit.LENGTH_SPLIT,
                    split_variation=split,
                    input_lang=self.input_lang,
                    output_lang=self.output_lang,
                    train=False
                )

                trainer = Seq2SeqTrainer.Seq2SeqTrainer(model=model,args=self.train_args,test_dataset=test_dataset,)
                metrics = trainer.evaluate()
                if split not in results:
                    results[split] = [metrics['eval_accuracy']]
                else:
                    results[split].append(metrics['eval_accuracy'])

        # Plot results
        self.plot_bar_chart(
            results=results,
            x_label=x_label,
            y_label='Accuracy on new commands (%)',
            save_path=f'plots/length-generalization-{self.run_type}.png'
        )


    def inspect_greedy_search(self):
        results = []

        for i, model in enumerate(self.saved_models):

            model.eval()

            oracle_best = []  # number of times oracle is better than greedy search
            with torch.no_grad():
                for input, target in tqdm(self.test_dataset, total=len(self.test_dataset), leave=False, desc="Evaluating"):
                    input_tensor, target_tensor = self.test_dataset.convert_to_tensor(input, target)

                    max_length = target_tensor.size(1)

                    _, greedy_prob = model.predict(input_tensor, max_length=100)

                    _, oracle_prob = model.predict(input_tensor, oracle_target=target_tensor)

                    oracle_best.append(oracle_prob > greedy_prob)

            results.append(oracle_best)


        # Calcate average amount of oracle search being greater than truth
        pct_oracle_best = [sum(data) / len(data) for data in results]
        avg_oracle_best = sum(pct_oracle_best) / len(pct_oracle_best)
        print(f'Average amount of oracle search being greater than truth: {avg_oracle_best}')

def main():

    train_args = config.paper_train_args

    Experiment2(
        model=RNNSeq2Seq.RNNSeq2Seq(),
        model_config=config.overall_best_config,
        train_args=train_args,
        run_type='overall_best',
        n_runs=1,
    ).run()




if __name__ == '__main__':
    main()
