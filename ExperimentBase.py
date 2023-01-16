import ScanDataset
import wandb
from matplotlib import pyplot as plt
import numpy as np
import Seq2SeqTrainer
import Seq2SeqModel
import torch
from typing import List, Dict, Tuple, Union, Optional
from CustomTrainerCallback import CustomTrainerCallback
import math


class ExperimentBase:
    def __init__(
        self,
        model: Seq2SeqModel.Seq2SeqModel,
        model_config: Seq2SeqModel.Seq2SeqModelConfig,
        train_args: Seq2SeqTrainer.Seq2SeqTrainingArguments,
        run_type: str,
        n_runs: int,
        split: ScanDataset.ScanSplit,
        split_variation: Union[str, List] = None,
        criterion: torch.nn.Module = None,
    ):
        self.model = model
        self.model_config = model_config
        self.train_args = train_args
        self.run_type = run_type
        self.n_runs = n_runs
        self.split = split
        self.split_variation = split_variation
        self.criterion = criterion

        self.load_data()

        self.model = model.from_config(self.model_config)

        self.saved_models = []

    def load_data(self):

        self.input_lang = ScanDataset.Lang()
        self.output_lang = ScanDataset.Lang()

        self.train_dataset = ScanDataset.ScanDataset(
            input_lang=self.input_lang,
            output_lang=self.output_lang,
            split=self.split,
            split_variation=self.split_variation,
            train=True,
        )
        self.test_dataset = ScanDataset.ScanDataset(
            input_lang=self.input_lang,
            output_lang=self.output_lang,
            split=self.split,
            split_variation=self.split_variation,
            train=False,
        )

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
                criterion=self.criterion,
                callbacks=[CustomTrainerCallback(run_index=i, run_type=self.run_type)],
            )

            model = trainer.train(evaluate_after=False)
            self.saved_models.append(model)
            metrics = trainer.evaluate()
            accuracies.append(metrics["eval_accuracy"])

        avg_accuracy = np.mean(accuracies)

        if self.train_args.log_wandb:
            wandb.log({f"avg_accuracy": avg_accuracy})

        print(f"Average accuracy: {avg_accuracy}")

    def length_generalization(
        self,
        split: ScanDataset.ScanSplit,
        splits: List[int],
        x_label: str = "",
        plot_title: str = "",
    ):

        results: Dict[int, List] = {}

        for i, model in enumerate(self.saved_models):

            # Evaluate on various lengths
            for split in splits:
                test_dataset = ScanDataset.ScanDataset(
                    split=self.split,
                    split_variation=split,
                    input_lang=self.input_lang,
                    output_lang=self.output_lang,
                    train=False,
                )

                trainer = Seq2SeqTrainer.Seq2SeqTrainer(
                    model=model,
                    args=self.train_args,
                    test_dataset=test_dataset,
                )
                metrics = trainer.evaluate()
                if split not in results:
                    results[split] = [metrics["eval_accuracy"]]
                else:
                    results[split].append(metrics["eval_accuracy"])

        # Plot results
        self.plot_bar_chart(
            results=results,
            x_label=x_label,
            y_label="Accuracy on new commands (%)",
            plot_title=plot_title,
            save_path=f"plots/{plot_title}-{self.run_type}.png",
        )

    def plot_bar_chart(
        self,
        results: Dict[int, List[float]] = {},
        x_label: str = "",
        y_label: str = "",
        plot_title: str = "",
        save_path: Optional[str] = None,
    ):
        # Average results
        mean_results = {}
        for split, result in results.items():
            mean_results[split] = sum(result) / len(result)

        # Calculate 1 SEM
        sem_results = {}
        for split, result in results.items():
            sem_results[split] = np.std(result) / math.sqrt(len(result))

        # Plot bar chart
        fig, ax = plt.subplots()
        ax.bar(
            list(results.keys()),
            list(mean_results.values()),
            align="center",
            yerr=list(sem_results.values()),
            capsize=5,
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        # TODO: figure out how to set x axis labels to exactly 'splits'
        # plt.xticks()
        ax.set_ylim(bottom=0, top=1)

        # if self.train_args.log_wandb:
        #     wandb.log({plot_title: fig})

        if save_path:
            plt.savefig(save_path)
