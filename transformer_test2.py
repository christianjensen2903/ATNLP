import scan_dataset
import torch
import numpy as np

# from transformer import Seq2SeqTransformer, Seq2SeqTransformerConfig

from Seq2SeqTransformer import Seq2SeqTransformer, Seq2SeqTransformerConfig
import Seq2SeqTrainer
import config
import wandb


input_lang = scan_dataset.Lang()
output_lang = scan_dataset.Lang()

dataset = scan_dataset.ScanDataset(
    split=scan_dataset.ScanSplit.SIMPLE_SPLIT,
    input_lang=input_lang,
    output_lang=output_lang,
    train=True,
)

test_dataset = scan_dataset.ScanDataset(
    split=scan_dataset.ScanSplit.SIMPLE_SPLIT,
    input_lang=input_lang,
    output_lang=output_lang,
    train=False,
)

SRC_VOCAB_SIZE = input_lang.n_words
TGT_VOCAB_SIZE = output_lang.n_words
EMB_SIZE = 128
NHEAD = 4
FFN_HID_DIM = 128
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2


n_iters = 10

transformer_config = Seq2SeqTransformerConfig(
    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,
    emb_size=EMB_SIZE,
    nhead=NHEAD,
    input_vocab_size=SRC_VOCAB_SIZE,
    output_vocab_size=TGT_VOCAB_SIZE,
    dim_feedforward=FFN_HID_DIM,
    dropout=0.1,
    pad_index=3,
    eos_index=1,
    sos_index=0,
)

train_args = config.paper_train_args

train_args.batch_size = BATCH_SIZE

if train_args.log_wandb:
    wandb.init(project="test", entity="atnlp", reinit=True, tags=["transformer"])

criterion = torch.nn.CrossEntropyLoss(ignore_index=3)


model = Seq2SeqTransformer(transformer_config)

trainer = Seq2SeqTrainer.Seq2SeqTrainer(
    model=model,
    args=train_args,
    train_dataset=dataset,
    test_dataset=test_dataset,
    criterion=criterion,
)

trainer.train()

metrics = trainer.evaluate()
print(metrics)

# for input, target in test_dataset:
#     input_tensor, target_tensor = test_dataset.convert_to_tensor(input, target)

#     pred = model.predict(input_tensor)
#     print(pred)
#     print(target_tensor)
