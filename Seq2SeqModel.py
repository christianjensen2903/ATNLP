import torch.nn as nn
import wandb
import torch
from dataclasses import dataclass
import pickle
import os

@dataclass
class Seq2SeqModelConfig():
    pad_index: int = None
    sos_index: int = None
    eos_index: int = None
    input_vocab_size: int = None
    output_vocab_size: int = None

# Abstract class for seq2seq models
class Seq2SeqModel(nn.Module):
    def __init__(self, config: Seq2SeqModelConfig = None):
        super(Seq2SeqModel, self).__init__()
        if config:
            self.from_config(config)

    def from_config(self, config: Seq2SeqModelConfig):
        self.pad_index = config.pad_index
        self.sos_index = config.sos_index
        self.eos_index = config.eos_index
        self.input_vocab_size = config.input_vocab_size
        self.output_vocab_size = config.output_vocab_size
        
        # Initialize weights
        for layers in self.children():
            for layer in layers:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()



    def from_pretrained(self, path: str, from_wandb, wandb_name = None):
        raise NotImplementedError

    def forward(self, input, target):
        raise NotImplementedError

    def predict(self, input, max_length=100):
        raise NotImplementedError

    def save(self, path: str, log_wandb: bool = False, wandb_name: str = None):
        
        os.makedirs(path, exist_ok=True)
        path += '/models.sav'
        torch.save(self.state_dict(), path)
        # pickle.dump(self, open(path, 'wb'))
        if log_wandb:
            wandb_name = wandb_name if wandb_name else 'model-checkpoint.sav'
            artifact = wandb.Artifact(wandb_name, type='model')
            artifact.add_file(path)
            wandb.log_artifact(artifact)


    def load(self, path: str, from_wandb, wandb_name = None):
        if from_wandb and wandb_name:
            artifact = wandb.use_artifact(wandb_name)
            artifact_dir = artifact.download()
            self.load_state_dict(torch.load(artifact_dir))
        else:
            self.load_state_dict(torch.load(path))
