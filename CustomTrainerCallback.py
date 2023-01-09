import Seq2SeqTrainer
import wandb

class CustomTrainerCallback(Seq2SeqTrainer.TrainerCallback):
    """
    Custom trainer callback to log experiments to wandb
    """

    def __init__(self, run_index: int, run_type: str):
        self.run_index = run_index
        self.run_type = run_type

    def _run_to_string(self):
        return f"{self.run_type}-{self.run_index}"


    def on_step_end(self, train_args: Seq2SeqTrainer.Seq2SeqTrainingArguments, state: Seq2SeqTrainer.TrainerState, **kwargs):
        """
        Called at the end of each training step.
        """
        super().on_step_end(train_args, state, **kwargs)
        # Add run type and index in front of each log
        log = state.log_history[-1]
        log = {f"{self._run_to_string()}/{key}": value for key, value in log.items()}
        if train_args.log_wandb:
            wandb.log(log)

    def on_save(self, train_args: Seq2SeqTrainer.Seq2SeqTrainingArguments, state: Seq2SeqTrainer.TrainerState, **kwargs):
        super().on_save(train_args, state, **kwargs)
        if train_args.log_wandb:
            artifact = wandb.Artifact(f'{self._run_to_string()}-checkpoint.sav', type='model')
            artifact.add_file(train_args.output_dir + "-checkpoint.sav")
            wandb.log_artifact(artifact)

