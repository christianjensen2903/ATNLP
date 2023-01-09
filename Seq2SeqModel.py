import torch.nn as nn
import wandb
import torch
from dataclasses import dataclass

@dataclass
class Seq2SeqModelConfig():
    pad_index: int
    sos_index: int
    eos_index: int

# Abstract class for seq2seq models
class Seq2SeqModel(nn.Module):
    def __init__(self, config: Seq2SeqModelConfig):
        super(Seq2SeqModel, self).__init__()
        self.from_config(config)

    def from_config(self, config: Seq2SeqModelConfig):
        self.pad_index = config.pad_index
        self.sos_index = config.sos_index
        self.eos_index = config.eos_index

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
        torch.save(self.state_dict(), path)
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


        