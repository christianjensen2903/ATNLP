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
import math

class ExperimentBase:
    def __init__(self,
                model: Seq2SeqModel.Seq2SeqModel,
                model_config: Seq2SeqModel.Seq2SeqModelConfig,
                train_args: Seq2SeqTrainer.Seq2SeqTrainingArguments,
                run_type: str,
                n_runs: int,
                split: scan_dataset.ScanSplit
                 ):
        self.model = model
        self.model_config = model_config
        self.train_args = train_args
        self.run_type = run_type
        self.n_runs = n_runs
        self.split = split

        self.load_data()

        self.model = model.from_config(self.model_config)

        self.saved_models = []
        


    def load_data(self):

        self.input_lang = scan_dataset.Lang()
        self.output_lang = scan_dataset.Lang()

        self.train_dataset = scan_dataset.ScanDataset(input_lang = self.input_lang, output_lang = self.output_lang, split = self.split, train = True)
        self.test_dataset = scan_dataset.ScanDataset(input_lang = self.input_lang, output_lang = self.output_lang, split = self.split, train = False)

        self.model_config.input_vocab_size = self.input_lang.n_words
        self.model_config.output_vocab_size = self.output_lang.n_words
        self.model_config.sos_index = self.output_lang.sos_index
        self.model_config.eos_index = self.output_lang.eos_index
        self.model_config.pad_index = self.output_lang.pad_index

    def run(self):
        """Run experiment"""
        pass


    def train_models(self):
        """Train and evaluate models"""

        accuracies = []

        for i in range(self.n_runs):

            # Reset model
            model = self.model.copy().reset_model()

            # Train and evaluate model
            trainer = Seq2SeqTrainer.Seq2SeqTrainer(
                model=model,
                args=self.train_args,
                train_dataset=self.train_dataset,
                test_dataset=self.test_dataset,
                callbacks=[CustomTrainerCallback(run_index=i, run_type=self.run_type)],
            )
            
            model = trainer.train(evaluate_after=False)
            self.saved_models.append(model)
            metrics = trainer.evaluate()
            accuracies.append(metrics['eval_accuracy'])

        print(f'Average accuracy for {self.run_type}: {np.mean(accuracies)}')


    def plot_bar_chart(self, results: Dict[int, List[float]] = {}, x_label: str = '', y_label: str = '', plot_title: str = '',
                   log_wandb: bool = False, save_path: Optional[str] = None):
        # Average results
        mean_results = {}
        for split, result in results.items():
            mean_results[split] = sum(result) / len(result)

        # Calculate 1 SEM
        sem_results = {}
        for split, result in results.items():
            sem_results[split] = np.std(result) / math.sqrt(len(result))
        

        # Plot bar chart
        plt.bar(list(results.keys()), list(mean_results.values()), align='center', yerr=list(sem_results.values()), capsize=5)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        # TODO: figure out how to set x axis labels to exactly 'splits'
        # plt.xticks()
        plt.ylim((0., 1.))

        if log_wandb:
            wandb.log({plot_title: plt})

        if save_path:
            plt.savefig(save_path)





    