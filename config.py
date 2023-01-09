from dataclasses import asdict, dataclass, field, fields
from Seq2SeqTrainer import Seq2SeqTrainer, Seq2SeqTrainingArguments


@dataclass
class RNNConfig():
    hidden_size: int
    n_layers: int
    dropout: float
    attention: bool = False
    rnn_type: str = "RNN"
    teacher_forcing_ratio: float = 0.5


overall_best = RNNConfig(
    hidden_size=200,
    n_layers=2,
    dropout=0.2,
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





