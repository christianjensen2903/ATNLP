import numpy as np
import torch
import torch.nn as nn
from torch import optim
import random
from tqdm import tqdm
import helper
import scan_dataset
import wandb

# TODO: Make encoder decoder into a model class so the training can 
# accommodate transformer and lstm

teacher_forcing_ratio = .5

def train_iteration(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
                    device='cpu'):
    """A single training iteration."""
    # Reset the gradients and loss
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0

    target_length = target_tensor.size(1)
    batch_size = target_tensor.size(0)

    # Encode the input
    encoder_hidden, all_encoder_hidden = encoder(input_tensor)

    # Prepare the initial decoder input
    decoder_input = torch.tensor([[scan_dataset.SOS_token]* batch_size], device=device).transpose(0, 1)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    for di in range(target_length):

        # Decode next token
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, all_encoder_hidden)

        # Calculate loss
        loss += criterion(decoder_output, target_tensor[:, di])

        # If teacher forcing is used, the next input is the target
        # Otherwise, the next input is the output with the highest probability
        if use_teacher_forcing:
            decoder_input = target_tensor[:, di].unsqueeze(1)
        else:
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.detach()  # detach from history as input

    loss.backward()
    
    nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
    nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train(dataset, encoder, decoder, n_iters, device='cpu', print_every=1000, plot_every=100, learning_rate=1e-3,
          verbose=False, plot=False, log_wandb=False, batch_size=1):
    encoder.train()
    decoder.train()

    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss(ignore_index=scan_dataset.PAD_token)

    for iteration in tqdm(range(1, n_iters + 1), total=n_iters, leave=False, desc="Training"):
        random_batch = np.random.choice(len(dataset), batch_size)
        X, y = dataset[random_batch]

        input_tensor, target_tensor = dataset.convert_to_tensor(X, y)

        loss = train_iteration(input_tensor.to(device), target_tensor.to(device), encoder, decoder, encoder_optimizer,
                               decoder_optimizer, criterion, device=device)
        print_loss_total += loss
        plot_loss_total += loss

        if iteration % print_every == 0 and verbose:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%d (%d%%): %.4f' % (iteration, iteration / n_iters * 100, print_loss_avg))

        if iteration % plot_every == 0:

            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)

            if log_wandb:
                wandb.log({"avg_loss": plot_loss_avg})
            plot_loss_total = 0

    if plot:
        helper.show_plot(plot_losses)

    return encoder, decoder


def evaluate(dataset, encoder, decoder, max_length, device='cpu', verbose=False):
    encoder.eval()
    decoder.eval()
    
    n_correct = [] # number of correct predictions
    
    with torch.no_grad():
        for input_tensor, target_tensor in tqdm(dataset, total=len(dataset), leave=False, desc="Evaluating"):
            input_tensor, target_tensor = dataset.convert_to_tensor(input_tensor, target_tensor)

            pred = []

            encoder_hidden, all_encoder_hidden = encoder(input_tensor)
            decoder_input = torch.tensor([[scan_dataset.SOS_token]], device=device)

            decoder_hidden = encoder_hidden
            
            target_length = target_tensor.size(0)

            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, all_encoder_hidden)

                topv, topi = decoder_output.topk(1)
                decoder_input = topi.detach()  # detach from history as input

                pred.append(decoder_input.item())

                if decoder_input.item() == scan_dataset.EOS_token:
                    break

            pred = np.array(pred)
            ground_truth = target_tensor.detach().cpu().numpy().squeeze()

            if len(pred) == len(ground_truth):
                n_correct.append(np.all(pred == ground_truth))
            else:
                n_correct.append(0)

    accuracy = np.mean(n_correct)
    if verbose:
        print("Accuracy", accuracy)
        
    encoder.train()
    decoder.train()

    return accuracy


def oracle_eval(dataset, encoder, decoder, device='cpu', verbose=False):
    encoder.eval()
    decoder.eval()

    n_correct = []  # number of correct predictions

    with torch.no_grad():
        for input_tensor, target_tensor in tqdm(dataset, total=len(dataset), leave=False, desc="Evaluating"):
            input_tensor, target_tensor = dataset.convert_to_tensor(input_tensor, target_tensor)

            pred = []

            encoder_hidden, all_encoder_hidden = encoder(input_tensor)
            decoder_input = torch.tensor([[scan_dataset.SOS_token]], device=device)

            decoder_hidden = encoder_hidden

            target_length = target_tensor.size(0)

            for di in range(target_length-1):
                
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, all_encoder_hidden)

                topv, topi = decoder_output.topk(1)
                decoder_input = topi.detach()  # detach from history as input

            
                if decoder_input.item() == scan_dataset.EOS_token:
                    topv, topi = decoder_output.topk(2)
                    decoder_input = topi[:, 1].detach().unsqueeze(0)  # detach from history as input
                    
                pred.append(decoder_input.squeeze().item())

            pred = np.array(pred)
            ground_truth = target_tensor.detach().cpu().numpy().squeeze()[:-1]

            if len(pred) == len(ground_truth):
                n_correct.append(np.all(pred == ground_truth))
            else:
                n_correct.append(0)

    accuracy = np.mean(n_correct)
    if verbose:
        print("Accuracy", accuracy)
    
    return accuracy
