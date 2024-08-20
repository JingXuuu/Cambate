import os

# Hyperparameters
input_dim = 500  # Dimension of input features (dimension of c3d features)
vocab_size = 50257  # Size of the vocabulary (gpt2)
hidden_dim = 768  # Hidden dimension size (encoder output to decoder)
n_mamba_layers = 12
n_transformer_layers = 4  # Number of Transformer decoder layers
n_heads = 8  # Number of attention heads
dropout = 0.1  # Dropout rate
sequence_length = 64  # Assuming a fixed sequence length (number of c3d features per caption)
batch_size = 240  # Batch size (240 is max for 80G GPU)
num_epochs = 10  # Number of epochs

# path to data
dir = './data/activitynet-captions'
train_id_path = os.path.join(dir, 'captions/train_ids.json')
train_caption_path = os.path.join(dir, 'captions/train.json')
test_id_path = os.path.join(dir, 'captions/val_ids.json')
test_caption_path = os.path.join(dir, 'captions/val_1.json')
feature_folder = os.path.join(dir, 'c3d')

tokenizer_path = "./gpt2"