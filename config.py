from dataclasses import asdict, dataclass, field, fields
from Seq2SeqTrainer import Seq2SeqTrainer, Seq2SeqTrainingArguments
from RNNSeq2Seq import RNNSeq2SeqConfig
from Seq2SeqTransformer import Seq2SeqTransformerConfig


overall_best_config = RNNSeq2SeqConfig(
    hidden_size=200,
    n_layers=2,
    dropout_p=0.2,
    attention=False,
    rnn_type="LSTM",
    teacher_forcing_ratio=0.5,
)

paper_train_args = Seq2SeqTrainingArguments(
    batch_size=1,
    n_iter=100000,
    print_every=10000,
    log_every=100,
    clip_grad=5.0,
    log_wandb=False,
    output_dir="checkpoints",
)

transformer_config = Seq2SeqTransformerConfig(
    nhead=8,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=256,
    emb_size=256,
    dropout=0.1,
)


transformer_train_args = Seq2SeqTrainingArguments(
    batch_size=128,
    n_iter=1000,
    print_every=1000,
    log_every=100,
    clip_grad=5.0,
    log_wandb=False,
    output_dir="checkpoints",
)
