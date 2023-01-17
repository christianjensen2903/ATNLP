import wandb
from Experiment1 import Experiment1
import config
import torch
import Seq2SeqTransformer
from Seq2SeqTrainer import Seq2SeqTrainer, Seq2SeqTrainingArguments


def main():
    # Use the wandb.init() API to generate a background process
    # to sync and log data as a Weights and Biases run.
    run = wandb.init()

    # note that we define values from `wandb.config` instead of
    # defining hard values
    bs = wandb.config.batch_size
    n_iter = wandb.config.n_iter
    nhead = wandb.config.nhead
    num_encoder_layers = wandb.config.num_encoder_layers
    num_decoder_layers = wandb.config.num_decoder_layers
    dim_feedforward = wandb.config.dim_feedforward
    emb_size = wandb.config.emb_size
    dropout = wandb.config.dropout
    num_beams = wandb.config.num_beams

    transformer_config = Seq2SeqTransformer.Seq2SeqTransformerConfig(
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        emb_size=emb_size,
        dropout=dropout,
        num_beams=num_beams,
    )

    transformer_train_args = Seq2SeqTrainingArguments(
        batch_size=bs,
        n_iter=n_iter,
        print_every=1000,
        log_every=100,
        clip_grad=5.0,
        log_wandb=True,
        output_dir="checkpoints",
    )

    model = Seq2SeqTransformer.Seq2SeqTransformer()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=3)

    Experiment1(
        model=model,
        model_config=transformer_config,
        train_args=transformer_train_args,
        run_type="transformer",
        n_runs=1,
        criterion=criterion,
    ).run()


# 🐝 Step 2: Define sweep config
sweep_configuration = {
    "method": "bayes",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "avg_accuracy"},
    "parameters": {
        "batch_size": {"values": [32, 64, 128]},
        "n_iter": {"values": [1000, 10000, 20000]},
        "nhead": {"values": [1, 2, 4, 8, 16]},
        "num_encoder_layers": {"values": [1, 2, 3]},
        "num_decoder_layers": {"values": [1, 2, 3]},
        "emb_size": {"values": [32, 64, 128, 256]},
        "dim_feedforward": {"values": [32, 64, 128, 256]},
        "dropout": {"values": [0, 0.1, 0.2, 0.3, 0.4, 0.5]},
        "num_beams": {"max": 10, "min": 1},
    },
}


# 🐝 Step 3: Initialize sweep by passing in config
# sweep_id = wandb.sweep(sweep=sweep_configuration, project="transformer-sweep")
sweep_id = "b7sc4l8t"

while True:
    # 🐝 Step 4: Call to `wandb.agent` to start a sweep
    wandb.agent(
        sweep_id, function=main, count=1, project="transformer-sweep", entity="atnlp"
    )
