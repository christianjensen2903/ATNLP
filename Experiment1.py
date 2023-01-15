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
import config
from ExperimentBase import ExperimentBase


class Experiment1(ExperimentBase):
    def __init__(
        self,
        model: Seq2SeqModel.Seq2SeqModel,
        model_config: Seq2SeqModel.Seq2SeqModelConfig,
        train_args: Seq2SeqTrainer.Seq2SeqTrainingArguments,
        run_type: str,
        n_runs: int,
    ):

        self.split = scan_dataset.ScanSplit.SIMPLE_SPLIT
        super().__init__(
            model, model_config, train_args, run_type, n_runs, split=self.split
        )

    def run(self):
        self.train_models()
        self.length_generalization(
            split=self.split,
            splits=["p1", "p2", "p4", "p8", "p16", "p32", "p64"],
            x_label="Percent of commands used for training",
            plot_title=f"training_pct_accuracy",
        )


def main():
    train_args = config.paper_train_args

    for model_type in ["overall_best", "experiment_best"]:

        if model_type == "overall_best":
            model = RNNSeq2Seq.RNNSeq2Seq()
            model_config = config.overall_best_config
        elif model_type == "experiment_best":
            model = RNNSeq2Seq.RNNSeq2Seq()
            model_config = RNNSeq2Seq.RNNSeq2SeqConfig(
                hidden_size=200,
                n_layers=2,
                dropout_p=0,
                attention=False,
                rnn_type="LSTM",
                teacher_forcing_ratio=0.5,
            )

        # Initialize wandb

        if train_args.log_wandb:
            wandb.init(
                project="experiment-1",
                entity="atnlp",
                config=train_args,
                reinit=True,
                tags=["experiment-1", model_type],
            )

        Experiment1(
            model=model,
            model_config=model_config,
            train_args=train_args,
            run_type=model_type,
            n_runs=1,
        ).run()


if __name__ == "__main__":
    main()
