import Seq2SeqTrainer
import wandb

class CustomTrainerCallback(Seq2SeqTrainer.TrainerCallback):
    """
    Custom trainer callback to log experiments to wandb
    """

    def __init__(self, wandb_run: wandb.Run, run_index: int, run_type: str):
        self.wandb_run = wandb_run
        self.run_index = run_index
        self.run_type = run_type

    def _run_to_string(self):
        return f"{self.run_type}-{self.run_index}"


    def on_train_end(self, state: Seq2SeqTrainer.TrainerState):
        """
        Save model after training
        """
        self.save(f'saved_models/experiment_2/run_{run_index}.sav', wandb_run=wandb_run, wandb_name=f'run_{run_index}')


    def on_step_end(self, state: Seq2SeqTrainer.TrainerState):
        """
        Called at the end of each training step.
        """
        # Add run type and index in front of each log
        log = state.log_history[-1]
        log = {f"{self._run_to_string()}/{key}": value for key, value in log.items()}

        if self.wandb_run is not None:
            self.wandb_run.log(log)