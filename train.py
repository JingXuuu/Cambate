import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from transformers import GPT2Tokenizer # type: ignore
from torch.nn.parallel import DataParallel

from encoder import MambaEncoder
from decoder import TransformerDecoder
from dataset import create_dataloaders
import config

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,5,6"
multi_gpu = torch.cuda.device_count() > 1

# select device
device = torch.device('cuda')


if __name__ == '__main__':
    ################################ Hyperparameters ################################
    input_dim = config.input_dim  # Dimension of input features (dimension of c3d features)
    vocab_size = config.vocab_size  # Size of the vocabulary (gpt2)
    hidden_dim = config.hidden_dim  # Hidden dimension size (encoder output to decoder)
    n_mamba_layers = config.n_mamba_layers
    n_transformer_layers = config.n_transformer_layers  # Number of Transformer decoder layers
    n_heads = config.n_heads  # Number of attention heads
    dropout = config.dropout  # Dropout rate
    sequence_length = config.sequence_length  # Assuming a fixed sequence length (number of c3d features per caption)
    batch_size = config.batch_size  # Batch size
    num_epochs = config.num_epochs  # Number of epochs

    # path to data
    dir = config.dir
    train_id_path = config.train_id_path
    train_caption_path = config.train_caption_path
    test_id_path = config.test_id_path
    test_caption_path = config.test_caption_path
    feature_folder = config.feature_folder

    # Instantiate models
    encoder = MambaEncoder(
        input_dim,  # Assuming input features have dimension 1024
        d_model=hidden_dim,
        n_layer=n_mamba_layers,
        d_intermediate=hidden_dim * 4,
        sequence_length=sequence_length
    ).to(device)

    decoder = TransformerDecoder(
        vocab_size, 
        hidden_dim, 
        n_transformer_layers,
        n_heads, 
        dropout
    ).to(device)


    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config.tokenizer_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id 
    ################################ Hyperparameters ################################

    if multi_gpu:
        encoder = DataParallel(encoder)
        decoder = DataParallel(decoder)

    # Loss, optimizer, scheduler (default values)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # data loader
    train_loader, test_loader = create_dataloaders(train_id_path, train_caption_path, test_id_path, test_caption_path, feature_folder, batch_size)

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        encoder.train()
        decoder.train()
        
        for i, (input_features, captions) in enumerate(tqdm(train_loader)):
            input_features = input_features[0].to(device)
            # tokenize captions, captions is a list of captions
            # add bos and eos token to the end of each caption 
            # pad all captions to the same length
            captions = [tokenizer.encode(f"{tokenizer.bos_token}{caption[0]}{tokenizer.eos_token}") for caption in captions]
            max_len = max([len(caption) for caption in captions])
            captions = [caption + [tokenizer.pad_token_id] * (max_len - len(caption)) for caption in captions]
            captions = torch.tensor(captions).to(device)
            
            # Forward pass
            encoder_output = encoder(input_features)
        
            # Prepare decoder inputs, create shifted target
            tgt_input = captions[:, :-1]
            tgt_output = captions[:, 1:]

            # Forward pass through decoder
            decoder_output = decoder(encoder_output, tgt_input)
            
            # Compute loss
            loss = criterion(decoder_output.reshape(-1, vocab_size), tgt_output.reshape(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        
        # Log loss
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss.item()}")
        with open('./ckpt/loss.txt', 'a') as f:
            f.write(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss.item()}\n")

        # save model for checkpoint
        torch.save(encoder.state_dict(), f'./ckpt/encoder_{epoch + 1}.pt')
        torch.save(decoder.state_dict(), f'./ckpt/decoder_{epoch + 1}.pt')

    print("Training complete!")