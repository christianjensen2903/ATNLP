from dataclasses import asdict, dataclass, field, fields
from Seq2SeqTrainer import Seq2SeqTrainer, Seq2SeqTrainingArguments
from RNNSeq2Seq import RNNSeq2SeqConfig



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
    clip_grad=5.0,
    log_wandb=False,
    output_dir="checkpoints",

)





