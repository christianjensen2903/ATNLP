import Seq2SeqModel
import torch
from dataclasses import dataclass
import Seq2SeqDataset
import wandb
import numpy as np
from tqdm import tqdm


@dataclass
class Seq2SeqTrainingArguments():
    batch_size: int = 32
    n_iter: int = 1000
    n_runs: int = 1
    learning_rate: float = 0.001
    clip_grad: float = 1.0
    teacher_forcing_ratio: float = 0.5
    log_wandb: bool = False


class Seq2SeqTrainer():

    def __init__(
            self,
            model: Seq2SeqModel.Seq2SeqModel,
            args: Seq2SeqTrainingArguments = None,
            optimizer: torch.optim.Optimizer = None,
            criterion: torch.nn.Module = None,
            train_dataset: Seq2SeqDataset.Seq2SeqDataset = None,
            test_dataset: Seq2SeqDataset.Seq2SeqDataset = None,
            ):


        if args is None:
            args = Seq2SeqTrainingArguments()

        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        
        if criterion is None:
            assert(train_dataset is not None, "Train dataset must be provided if criterion is not provided")
            criterion = torch.nn.NLLLoss(ignore_index=train_dataset.output_lang.pad_index)

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.args = args
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def train(self, verbose=False, log_every=1000):
        """Train the model for n_iter iterations."""
        self.model.train()

        print_loss_total = 0

        for iteration in tqdm(range(1, self.args.n_iter + 1), total=self.args.n_iter, leave=False, desc="Training"):
            random_batch = np.random.choice(len(self.train_dataset), self.args.batch_size)
            X, y = self.train_dataset[random_batch]

            input_tensor, target_tensor = self.train_dataset.convert_to_tensor(X, y)

            loss = self.train_iteration(input_tensor, target_tensor)
            print_loss_total += loss

            if iteration % log_every == 0 and verbose:
                print_loss_avg = print_loss_total / log_every
                print_loss_total = 0
                print('%d (%d%%): %.4f' % (iteration, iteration / self.args.n_iter * 100, print_loss_avg))


    def train_iteration(self, input_tensor, target_tensor):
        """Train the model for a single iteration."""
        self.model.train()
        self.optimizer.zero_grad()

        loss = self.model(
            input_tensor.to(self.device),
            target_tensor.to(self.device),
            self.criterion
            )
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
        self.optimizer.step()

        return loss.item()

    
    def evaluate(self, verbose=False):
        """Evaluate the model on the test dataset."""
        self.model.eval()

        n_correct = []  # number of correct predictions

        with torch.no_grad():
            for input_tensor, target_tensor in tqdm(self.test_dataset, total=len(self.test_dataset), leave=False, desc="Evaluating"):
                input_tensor, target_tensor = self.test_dataset.convert_to_tensor(input_tensor, target_tensor)

                max_length = target_tensor.size(1)

                pred = self.model.predict(input_tensor, max_length=max_length)

                pred = pred.squeeze().cpu().numpy()
                ground_truth = target_tensor.numpy().squeeze()

                n_correct.append(np.all(pred == ground_truth))

        accuracy = np.mean(n_correct)

        if verbose:
            print("Accuracy", accuracy)

        return accuracy

                
