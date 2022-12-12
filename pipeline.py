import numpy as np
import torch
import torch.nn as nn
from torch import optim
import random
from tqdm import tqdm
import helper
import scan_dataset
import wandb



teacher_forcing_ratio = .5

def train_iteration(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device='cpu'):
    """A single training iteration."""
    # Reset the gradients and loss
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0

    # Encode the input
    encoder_hidden, encoder_hidden_all = encoder(input_tensor)

    # Prepare the initial decoder input
    decoder_input = torch.tensor([[scan_dataset.SOS_token]], device=device)

    if decoder.rnn_type == 'LSTM':
        decoder_hidden = encoder_hidden[0]
    else:
        decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    target_length = target_tensor.size(0)
    for di in range(target_length):
        # Decode next token
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_hidden_all)
        
        loss += criterion(decoder_output, target_tensor[di])

        # If teacher forcing is used, the next input is the target
        # Otherwise, the next input is the output with the highest probability
        if use_teacher_forcing:
            decoder_input = target_tensor[di]
        else:
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

        # If the decoder input is the EOS token, stop decoding
        if decoder_input.item() == scan_dataset.EOS_token:
            break

    loss.backward()
    
    nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
    nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length



def train(dataset, encoder, decoder, n_iters, device='cpu', print_every=1000, plot_every=100, learning_rate=1e-2, verbose = False, plot=False, log_wandb=False):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for iteration in tqdm(range(1, n_iters + 1), total=n_iters, leave=False, desc="Training"):
        X, y = dataset[random.randrange(len(dataset))]
        input_tensor, target_tensor = dataset.convert_to_tensor(X, y)

        loss = train_iteration(input_tensor.to(device), target_tensor.to(device), encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device=device)
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
            # print(input_tensor, target_tensor)
            input_tensor, target_tensor = dataset.convert_to_tensor(input_tensor, target_tensor)
            
            pred = []

            encoder_hidden, encoder_hidden_all = encoder(input_tensor.to(device))

            decoder_input = torch.tensor([[scan_dataset.SOS_token]], device=device)

            decoder_hidden = encoder_hidden
            
            target_length = target_tensor.size(0)

            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden, encoder_hidden_all)
                    
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

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
