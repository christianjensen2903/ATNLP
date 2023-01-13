import scan_dataset
import torch
import numpy as np
from transformer import Seq2SeqTransformer, Seq2SeqTransformerConfig


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

MAX_LENGTH = max(dataset.input_lang.max_length, dataset.output_lang.max_length)


SRC_VOCAB_SIZE = input_lang.n_words
TGT_VOCAB_SIZE = output_lang.n_words
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

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
)

n_iters = 10

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=3)

model = Seq2SeqTransformer(transformer_config)


def train_epoch(model, optimizer):
    model.train()
    losses = 0

    for _ in range(n_iters):
        random_batch = np.random.choice(len(dataset), BATCH_SIZE)
        X, y = dataset[random_batch]

        src, tgt = dataset.convert_to_tensor(X, y)

        # tgt_input = tgt[:-1, :]

        logits = model(src, tgt)

        optimizer.zero_grad()

        # tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / n_iters


print("Training model")
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
for epoch in range(10):
    loss = train_epoch(model, optimizer)
    print(f"Epoch {epoch} loss: {loss}")


# def evaluate(model):
#     model.eval()
#     losses = 0

#     val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
#     val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

#     for src, tgt in val_dataloader:
#         src = src.to(DEVICE)
#         tgt = tgt.to(DEVICE)

#         tgt_input = tgt[:-1, :]

#         src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

#         logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

#         tgt_out = tgt[1:, :]
#         loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
#         losses += loss.item()

#     return losses / len(val_dataloader)
