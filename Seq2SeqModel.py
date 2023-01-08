import torch.nn as nn

# Abstract class for seq2seq models
class Seq2SeqModel(nn.Module):
    def __init__(self, pad_index, sos_index, eos_index, **kwargs):
        super(Seq2SeqModel, self).__init__()
        self.pad_index = pad_index
        self.sos_index = sos_index
        self.eos_index = eos_index

    def forward(self, input, target):
        raise NotImplementedError

    def predict(self, input, max_length=100):
        raise NotImplementedError