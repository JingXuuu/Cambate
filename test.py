import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import numpy as np
import config
import json
import argparse

from encoder import MambaEncoder
from decoder import TransformerDecoder

from transformers import GPT2Tokenizer # type: ignore

# select device
device = torch.device('cuda:3')

# Load the checkpoint into the model
def load_ckpt(encoder, decoder, dir, epoch):
    # Load the state dictionaries
    encoder_state_dict = torch.load(os.path.join(dir, f"encoder_{epoch}.pt"))
    decoder_state_dict = torch.load(os.path.join(dir, f"decoder_{epoch}.pt"))

    # Remove the "module." prefix from the keys
    encoder_state_dict = {k.replace("module.", ""): v for k, v in encoder_state_dict.items()}
    decoder_state_dict = {k.replace("module.", ""): v for k, v in decoder_state_dict.items()}

    # Load the state dictionaries into the models
    encoder.load_state_dict(encoder_state_dict)
    decoder.load_state_dict(decoder_state_dict)

    # Move the models to the device
    encoder.to(device)
    decoder.to(device)


def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def load_feats(key, feature_folder):
    filepath = os.path.join(feature_folder, key + '.npy')
    return np.load(filepath)


def getitem(ids, captions, feature_folder, idx):
    key = str(ids[idx])
    captions_details = captions[key]
    
    # Validation check
    feature_file_path = os.path.join(feature_folder, key + '.npy')
    if not os.path.exists(feature_file_path):
        return None, None
    
    # load necessary data from captions
    feats = load_feats(key, feature_folder)
    duration = captions_details['duration']
    frames = feats.shape[0]

    captions = captions_details['sentences']
    gt_timestamps = captions_details['timestamps']
    gt_frames = (np.array(gt_timestamps) / duration * frames).astype(int)

    # Create [batch size, 64, feature dimension] batches, pad to make divisible by 64
    pad_len = (64 - frames % 64) % 64
    feats_padded = np.pad(feats, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
    frames_padded = feats_padded.shape[0]

    batch_size = (frames_padded // 64)
    feature_batches = np.zeros((batch_size, 64, feats.shape[1]))
    caption_batches = [""] * batch_size

    time = []
    for i in range(batch_size):
        start_idx = i * 64
        end_idx = (i + 1) * 64

        feature_batches[i] = feats_padded[start_idx:end_idx]

        # Assign captions to batches
        for caption, (start, end) in zip(captions, gt_frames):
            if start_idx <= start < end_idx or start_idx <= end < end_idx or (start < start_idx and end > end_idx):
                if caption_batches[i] == "":
                    caption_batches[i] = caption
                else:
                    caption_batches[i] += caption

        # calculate time for every caption
        time.append((start_idx, end_idx))

    time = np.array(time)
    time = (time / frames * duration)
    time[-1][-1] = duration # fix for the extra padding
    
    return feature_batches, caption_batches, time, key


# greedy decoding for simplicity
def generate_caption_greedy(encoder, decoder, features, max_length=100):
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        features = features.to(device)
        encoder_output = encoder(features)
        
        tgt_input = torch.tensor([tokenizer.bos_token_id]).unsqueeze(0).to(device)
        for i in range(max_length):
            output = decoder(encoder_output, tgt_input)
            output = output[:, -1, :]
            pred = output.argmax(1).item()
            tgt_input = torch.cat([tgt_input, torch.tensor([pred]).unsqueeze(0).to(device)], dim=1)
            if pred == tokenizer.eos_token_id:
                break

    return tgt_input.squeeze().cpu().numpy()

# beam search decoding
def generate_caption_beam(encoder, decoder, features, beam_width=5, max_length=100):
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        features = features.to(device)
        encoder_output = encoder(features)
        
        # Initialize the beams
        beams = [(torch.tensor([tokenizer.bos_token_id]).unsqueeze(0).to(device), 0)]
        
        for _ in range(max_length):
            new_beams = []
            for tgt_input, score in beams:
                output = decoder(encoder_output, tgt_input)
                output = output[:, -1, :]  # Take the last token prediction
                
                # Get the top-k tokens and their scores
                top_k_scores, top_k_tokens = torch.topk(output, beam_width, dim=-1)
                
                for i in range(beam_width):
                    token = top_k_tokens[0, i].item()
                    new_score = score + top_k_scores[0, i].item()
                    new_tgt_input = torch.cat([tgt_input, torch.tensor([[token]]).to(device)], dim=1)
                    
                    new_beams.append((new_tgt_input, new_score))
            
            # Select the top beam_width beams with the highest scores
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            
            # Check if any of the beams has generated the end token
            for tgt_input, score in beams:
                if tgt_input[0, -1].item() == tokenizer.eos_token_id:
                    return tgt_input.squeeze().cpu().numpy()
    
    # If no beam generates the end token, return the best sequence
    return beams[0][0].squeeze().cpu().numpy()





if __name__ == "__main__":
    # load everything

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


    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Video captioning')
    parser.add_argument('--video_idx', type=int, default=0, help='Video index')
    parser.add_argument('--feature_set', type=str, default='train', help='Feature set (train or test)')
    parser.add_argument('--epoch', type=int, default=8, help='Epoch to load')
    args = parser.parse_args()


    # choose checkpoint
    load_ckpt(encoder, decoder, './ckpt', args.epoch)

    # Choose dataset
    video_idx = args.video_idx
    feature_set = args.feature_set

    if feature_set == "train":
        ids = load_json(train_id_path)
        captions = load_json(train_caption_path)
    else:
        ids = load_json(test_id_path)
        captions = load_json(test_caption_path)

    # Get the features, captions, and timestamps
    # features: [batch size, 64, 500]
    # captions: [batch size]
    # time: [batch size, 2]
    # key: video id
    features, gt_captions, time, key = getitem(ids, captions, feature_folder, video_idx)
    features = torch.tensor(features).float().to(device)

    # clean the output file, print out the video link
    with open('output.txt', 'w') as f:
        f.write("")
    with open('output.txt', 'a') as f:
        f.write(f"Video ID: {key}\n")
        f.write(f"Video Link: https://www.youtube.com/watch?v={key[2:]}\n\n")
    print("download video here: https://www.ytube.com/watch?v=" + key[2:])

    # generate caption
    final_captions = []
    for i in range(features.shape[0]):
        beam_caption = generate_caption_beam(encoder, decoder, features[i].unsqueeze(0))[1:]

        # detokenize
        beam_caption = tokenizer.decode(beam_caption)

        # split into 3 sentences, keep the '.' at the end
        beam_caption = [sentence + '.' for sentence in beam_caption.split('.') if sentence][:3]
        
        # split the time into 3 parts, and pair with the caption
        # for example, time[i] = [0, 18], then split into 3 parts: [0, 6], [6, 12], [12, 18]
        num = 3
        start, end = time[i]
        interval = (end - start) / num
        time_intervals = [[start + j * interval, start + (j + 1) * interval] for j in range(num)]

        for j in range(num):
            result = {'timestamp': time_intervals[j], 'sentence': beam_caption[j]}
            final_captions.append(result)

        # output to a file
        with open('output.txt', 'a') as f:
            f.write(f"Timestamp: {time[i]}\n")
            f.write(f"Generated Caption {i + 1}: \n")
            for j in range(num):
                f.write(f"\t{beam_caption[j]}\n")
            f.write(f"Ground truth: {gt_captions[i]}\n\n")



    with open('visualization/results.json', 'w') as f:
        json.dump(final_captions, f)









