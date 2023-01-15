import scan_dataset
import torch
import numpy as np
from transformer import Seq2SeqTransformer, Seq2SeqTransformerConfig
import Seq2SeqTrainer
import config
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer


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
EMB_SIZE = 32
NHEAD = 2
FFN_HID_DIM = 32
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2


n_iters = 100

transformer_config = Seq2SeqTransformerConfig(
    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,
    emb_size=EMB_SIZE,
    nhead=NHEAD,
    input_vocab_size=SRC_VOCAB_SIZE,
    target_vocab_size=TGT_VOCAB_SIZE,
    dim_feedforward=FFN_HID_DIM,
    dropout=0.1,
    pad_index=3,
    eos_index=1,
    sos_index=0,
)


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss
logits = outputs.logits

# configuration = GPT2Config(
#     vocab_size=SRC_VOCAB_SIZE+TGT_VOCAB_SIZE,
#     n_positions=100,
#     n_ctx=100,
#     n_embd=EMB_SIZE,
#     n_layer=NUM_ENCODER_LAYERS,
#     n_head=NHEAD,
#     resid_pdrop=0.1,
#     embd_pdrop=0.1,
#     attn_pdrop=0.1,
#     layer_norm_epsilon=1e-5,
#     initializer_range=0.02,
#     summary_type="cls_index",
#     summary_use_proj=True,
#     summary_activation=None,
# )


# model = GPT2LMHeadModel(configuration)

# loss_fn = torch.nn.CrossEntropyLoss(ignore_index=3)


# def train_epoch(model, optimizer):
#     model.train()
#     losses = 0

#     for _ in range(n_iters):
#         random_batch = np.random.choice(len(dataset), BATCH_SIZE)
#         X, y = dataset[random_batch]

#         src, tgt = dataset.convert_to_tensor(X, y)

#         # tgt_input = tgt[:-1, :]

#         logits = model(input_ids=src, decoder_input_ids=tgt).logits

#         optimizer.zero_grad()

#         # tgt_out = tgt[1:, :]
#         loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
#         loss.backward()

#         optimizer.step()
#         losses += loss.item()

#     return losses / n_iters


# print("Training model")
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# for epoch in range(100):
#     loss = train_epoch(model, optimizer)
#     print(f"Epoch {epoch} loss: {loss}")
