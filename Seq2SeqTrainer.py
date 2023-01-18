import Seq2SeqModel
import torch
from dataclasses import dataclass, field
import Seq2SeqDataset
import wandb
import numpy as np
from tqdm import tqdm
from typing import List, Dict


@dataclass
class Seq2SeqTrainingArguments:
    batch_size: int = 32
    n_iter: int = 10000
    learning_rate: float = 0.001
    clip_grad: float = 1.0
    log_wandb: bool = False
    output_dir: str = None
    save_steps: int = None
    log_wandb: bool = False
    log_every: int = 10000
    print_every: int = 10000


@dataclass
class TrainerState:
    """
    Object containing the state of the trainer.
    """

    step: int = 0
    log_history: List[Dict[str, float]] = field(default_factory=list)


class TrainerCallback:
    def on_train_begin(
        self, train_args: Seq2SeqTrainingArguments, state: TrainerState, **kwargs
    ):
        """
        Called at the beginning of the training.
        """
        pass

    def on_train_end(
        self, train_args: Seq2SeqTrainingArguments, state: TrainerState, **kwargs
    ):
        """
        Called at the end of the training.
        """
        pass

    def on_step_begin(
        self, train_args: Seq2SeqTrainingArguments, state: TrainerState, **kwargs
    ):
        """
        Called at the beginning of each training step.
        """
        pass

    def on_step_end(
        self, train_args: Seq2SeqTrainingArguments, state: TrainerState, **kwargs
    ):
        """
        Called at the end of each training step.
        """
        pass

    def on_save(
        self, train_args: Seq2SeqTrainingArguments, state: TrainerState, **kwargs
    ):
        """
        Called when the model is saved.
        """
        pass

    def on_log(
        self, train_args: Seq2SeqTrainingArguments, state: TrainerState, **kwargs
    ):
        """
        Called when the training is logged.
        """
        pass


class Seq2SeqTrainer:
    def __init__(
        self,
        model: Seq2SeqModel.Seq2SeqModel,
        args: Seq2SeqTrainingArguments = None,
        optimizer: torch.optim.Optimizer = None,
        criterion: torch.nn.Module = None,
        train_dataset: Seq2SeqDataset.Seq2SeqDataset = None,
        test_dataset: Seq2SeqDataset.Seq2SeqDataset = None,
        callbacks: List[TrainerCallback] = [],
    ):

        if test_dataset is None:
            test_dataset = train_dataset

        if args is None:
            args = Seq2SeqTrainingArguments()

        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        if criterion is None:
            if train_dataset:
                criterion = torch.nn.NLLLoss(
                    ignore_index=train_dataset.output_lang.pad_index
                )
            else:
                criterion = torch.nn.NLLLoss()

        self.callbacks = callbacks
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.args = args
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(self.device)
        self.state = TrainerState()

    def train(self, verbose: bool = True, evaluate_after: bool = False):
        """Train the model for n_iter iterations."""
        self.model.train()

        print_loss_total = 0
        log_loss_total = 0

        # Training begin callback
        for callback in self.callbacks:
            callback.on_train_begin(state=self.state, train_args=self.args)

        for iteration in tqdm(
            range(1, self.args.n_iter),
            total=self.args.n_iter,
            leave=False,
            desc="Training",
        ):

            self.state.step = iteration
            # Step begin callback
            for callback in self.callbacks:
                callback.on_step_begin(state=self.state, train_args=self.args)

            random_batch = np.random.choice(
                len(self.train_dataset), self.args.batch_size
            )
            X, y = self.train_dataset[random_batch]

            input_tensor, target_tensor = self.train_dataset.convert_to_tensor(X, y)

            loss = self.train_iteration(input_tensor, target_tensor)
            print_loss_total += loss
            log_loss_total += loss

            if iteration % self.args.print_every == 0 and verbose:
                print_loss_avg = print_loss_total / self.args.log_every
                print_loss_total = 0
                print(
                    "%d (%d%%): %.4f"
                    % (iteration, iteration / self.args.n_iter * 100, print_loss_avg)
                )

            if iteration % self.args.log_every == 0:
                log_loss_avg = log_loss_total / self.args.log_every
                log_loss_total = 0
                if self.args.log_wandb:
                    wandb.log({"loss": log_loss_avg})

                self.state.log_history.append({"loss": log_loss_avg})
                for callback in self.callbacks:
                    callback.on_log(state=self.state, train_args=self.args)

            if (
                self.args.save_steps is not None
                and iteration % self.args.save_steps == 0
            ):
                self.model.save(self.args.output_dir)

            # Step end callback
            for callback in self.callbacks:
                callback.on_step_end(state=self.state, train_args=self.args)

        # Save after training
        if self.args.output_dir:
            self.model.save(self.args.output_dir)
            for callback in self.callbacks:
                callback.on_save(state=self.state, train_args=self.args)

        # Training end callback
        for callback in self.callbacks:
            callback.on_train_end(state=self.state, train_args=self.args)

        if evaluate_after:
            self.evaluate(verbose=verbose)

        return self.model

    def train_iteration(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor):
        """Train the model for a single iteration."""
        self.model.train()
        self.optimizer.zero_grad()

        input_tensor = input_tensor.to(self.device)
        target_tensor = target_tensor.to(self.device)
        outputs = self.model(input_tensor, target_tensor)

        loss = self.criterion(outputs.permute(0, 2, 1), target_tensor)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
        self.optimizer.step()
        return loss.item()

    def evaluate(self, verbose: bool = False):
        """Evaluate the model on the test dataset."""
        self.model.eval()
        iter = 0
        n_correct = []  # number of correct predictions
        with torch.no_grad():
            for input, target in tqdm(
                self.test_dataset,
                total=len(self.test_dataset),
                leave=False,
                desc="Evaluating",
            ):
                input_tensor, target_tensor = self.test_dataset.convert_to_tensor(
                    input, target
                )

                pred, _ = self.model.predict(input_tensor.to(self.device))

                pred = pred.squeeze().cpu().numpy()
                ground_truth = target_tensor.numpy().squeeze()
                if iter < 10:
                    print(pred, ground_truth)
                    print(pred.shape, ground_truth.shape)

                if iter > 3000:
                    break

                iter += 1

                n_correct.append(np.all(pred == ground_truth))

        accuracy = np.mean(n_correct)

        if verbose:
            print("Evaluation Accuracy", accuracy)

        return {"eval_accuracy": accuracy}
