import config
import ScanDataset
import RNNSeq2Seq
import torch
import wandb
import numpy as np
from tqdm import tqdm
import Seq2SeqTrainer
import Seq2SeqModel
import Seq2SeqTransformer
from typing import List, Dict, Tuple, Union, Optional
import config
from ExperimentBase import ExperimentBase


class Experiment3(ExperimentBase):
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

        super().__init__(
            model,
            model_config,
            train_args,
            run_type,
            n_runs,
            split=ScanDataset.ScanSplit.ADD_PRIM_SPLIT,
            split_variation=split_variation,
            criterion=criterion,
        )

    def run(self):
        self.train_models()


def main():

    for model_type in ["transformer"]:  # "overall_best", "experiment_best"
        for split_variation in ["jump", "turn_left"]:

            criterion = None
            if model_type == "overall_best":
                train_args = config.paper_train_args
                model = RNNSeq2Seq.RNNSeq2Seq()
                model_config = config.overall_best_config
            elif model_type == "experiment_best":
                train_args = config.paper_train_args
                model = RNNSeq2Seq.RNNSeq2Seq()
                model_config = RNNSeq2Seq.RNNSeq2SeqConfig(
                    hidden_size=100,
                    n_layers=1,
                    dropout_p=0.5,
                    attention=True,
                    rnn_type="GRU",
                    teacher_forcing_ratio=0.5,
                )
            elif model_type == "transformer":
                model = Seq2SeqTransformer.Seq2SeqTransformer()
                model_config = config.transformer_config
                train_args = config.transformer_train_args
                criterion = torch.nn.CrossEntropyLoss(ignore_index=3)

            # Initialize wandb
            if train_args.log_wandb:
                wandb.init(
                    project="experiment-3",
                    entity="atnlp",
                    config=train_args,
                    reinit=True,
                    tags=["experiment-3", model_type, split_variation],
                )

            Experiment3(
                model=model,
                model_config=model_config,
                train_args=train_args,
                run_type=model_type,
                n_runs=1,
                split_variation=split_variation,
                criterion=criterion,
            ).run()

            wandb.finish()


if __name__ == "__main__":
    main()
