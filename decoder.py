import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.nn import TransformerDecoder, TransformerDecoderLayer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_layers, n_heads, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_dim, n_heads, hidden_dim * 4, dropout),
            num_layers=n_layers
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
    
    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    
    def forward(self, encoder_output, tgt):
        # encoder_output: [Batch, Time, Hidden_dim]
        # tgt: [Batch, Tgt_len]
        tgt = tgt.int()
        
        # Add positional encoding to target
        tgt_emb = self.embedding(tgt) * math.sqrt(encoder_output.size(-1))
        tgt_emb = self.pos_encoding(tgt_emb.transpose(0, 1)).transpose(0, 1)

        # Generate target mask
        tgt_mask = self.generate_square_subsequent_mask(tgt_emb.size(1)).to(tgt_emb.device)
        
        # Forward pass through Transformer decoder
        output = self.transformer_decoder(
            tgt_emb.transpose(0, 1), # Transformer expects [Tgt_len, Batch, Hidden_dim]
            encoder_output.transpose(0, 1), # Transformer expects [Src_len, Batch, Hidden_dim]
            tgt_mask=tgt_mask
        )
        
        # Output projection
        output = self.fc_out(output.transpose(0, 1)) # Back to [Batch, Tgt_len, Vocab_size]
        
        return output