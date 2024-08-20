import torch

from mamba_ssm import Mamba
from encoder import MambaEncoder
from decoder import TransformerDecoder
import config
from torchinfo import summary


# test whether Mamba is runnable
batch, length, dim = 2, 64, 16
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=dim, # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
).to("cuda")
y = model(x)
assert y.shape == x.shape



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

# Instantiate models
encoder = MambaEncoder(
    input_dim,  # Assuming input features have dimension 1024
    d_model=hidden_dim,
    n_layer=n_mamba_layers,
    d_intermediate=hidden_dim * 4,
    sequence_length=sequence_length
).to("cuda")

decoder = TransformerDecoder(
    vocab_size, 
    hidden_dim, 
    n_transformer_layers,
    n_heads, 
    dropout
).to("cuda")
################################ Hyperparameters ################################

# Example usage
input_features = torch.randn(batch_size, sequence_length, input_dim).to("cuda")
output_features = encoder(input_features)
print("encoder output:", output_features.shape)  # Expected shape: (batch_size, sequence_length, hidden_dim)

# Forward pass through Transformer decoder
tgt = torch.randint(0, vocab_size, (batch_size, 10)).to("cuda")
output = decoder(output_features, tgt)
print("decoder output:", output.shape)  # Expected shape: (batch_size, tgt_len, vocab_size)

# # Print model summary
# summary(encoder, input_size=input_features.shape, device='cuda')
# summary(decoder, input_size=(output_features.shape, tgt.shape), device='cuda')










