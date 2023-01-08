import torch.nn as nn
import wandb
import torch

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

    def save(self, path: str, wandb_run = None, wandb_name = None):
        torch.save(self.state_dict(), path)
        if wandb_name and wandb_run:
            artifact = wandb.Artifact(wandb_name, type='model')
            artifact.add_file(path)
            wandb_run.log_artifact(artifact)


    def load(self, path: str, from_wandb, wandb_name = None):
        if from_wandb and wandb_name:
            artifact = wandb.use_artifact(wandb_name)
            artifact_dir = artifact.download()
            self.load_state_dict(torch.load(artifact_dir))
        else:
            self.load_state_dict(torch.load(path))


        