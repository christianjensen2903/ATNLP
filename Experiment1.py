import config
import ScanDataset
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
import Seq2SeqTransformer


class Experiment1(ExperimentBase):
    def __init__(
        self,
        model: Seq2SeqModel.Seq2SeqModel,
        model_config: Seq2SeqModel.Seq2SeqModelConfig,
        train_args: Seq2SeqTrainer.Seq2SeqTrainingArguments,
        run_type: str,
        n_runs: int,
        split: ScanDataset.ScanSplit = None,
        split_variation: Union[str, List] = None,
        criterion: torch.nn.Module = None,
    ):

        self.split = ScanDataset.ScanSplit.SIMPLE_SPLIT
        super().__init__(
            model,
            model_config,
            train_args,
            run_type,
            n_runs,
            split=self.split,
            split_variation=split_variation,
            criterion=criterion,
        )

    def run(self):
        self.train_models()
        # self.length_generalization(
        #     split=self.split,
        #     splits=["p1", "p2", "p4", "p8", "p16", "p32", "p64"],
        #     x_label="Percent of commands used for training",
        #     plot_title=f"training_pct_accuracy",
        # )


def main():

    for model_type in ["transformer"]:  # "overall_best", "experiment_best"

        criterion = None
        if model_type == "overall_best":
            model = RNNSeq2Seq.RNNSeq2Seq()
            train_args = config.paper_train_args
            model_config = config.overall_best_config
        elif model_type == "experiment_best":
            model = RNNSeq2Seq.RNNSeq2Seq()
            train_args = config.paper_train_args
            model_config = RNNSeq2Seq.RNNSeq2SeqConfig(
                hidden_size=200,
                n_layers=2,
                dropout_p=0,
                attention=False,
                rnn_type="LSTM",
                teacher_forcing_ratio=0.5,
            )
        elif model_type == "transformer":
            train_args = config.transformer_train_args
            model = Seq2SeqTransformer.Seq2SeqTransformer()
            model_config = config.transformer_config
            criterion = torch.nn.CrossEntropyLoss(ignore_index=3)

        # Initialize wandb
        if train_args.log_wandb:
            wandb.init(
                project="experiment-1",
                entity="atnlp",
                config=model_config,
                reinit=True,
                tags=["experiment-1", model_type],
            )

        Experiment1(
            model=model,
            model_config=model_config,
            train_args=train_args,
            run_type=model_type,
            n_runs=1,
            criterion=criterion,
        ).run()

        wandb.finish()


if __name__ == "__main__":
    main()
