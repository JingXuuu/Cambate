import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import GPT2Tokenizer # type: ignore

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def load_feats(key, feature_folder):
    filepath = os.path.join(feature_folder, key + '.npy')
    return np.load(filepath)

class VideoCaptionDataset(Dataset):
    def __init__(self, ids, captions, feature_folder, batch_size=128):
        self.ids = ids
        self.captions = captions
        self.feature_folder = feature_folder
        self.batch_size = batch_size
        
        # Prepare all data during initialization
        self.all_feature_batches, self.all_caption_batches = self.prepare_data()
    
    def prepare_data(self):
        # not a memory efficient way to store all data
        all_feature_batches = []
        all_caption_batches = []
        
        for idx in range(len(self.ids)):
            feature_batches, caption_batches = self.getitem(idx)
            if feature_batches is None or caption_batches is None:
                continue
            for fb, cb in zip(feature_batches, caption_batches):
                all_feature_batches.append(fb)
                all_caption_batches.append(cb)
        
        # Combine into larger batches
        large_feature_batches = []
        large_caption_batches = []
        
        num_batches = len(all_feature_batches)
        for i in range(0, num_batches, self.batch_size):
            if i + self.batch_size <= num_batches:
                large_feature_batch = np.array(all_feature_batches[i:i + self.batch_size])
                large_caption_batch = all_caption_batches[i:i + self.batch_size]
                
                # Ensure dimensions are correct
                assert large_feature_batch.shape == (self.batch_size, 64, large_feature_batch.shape[2])
                
                large_feature_batches.append(large_feature_batch)
                large_caption_batches.append(large_caption_batch)
        
        return large_feature_batches, large_caption_batches

    def getitem(self, idx):
        key = str(self.ids[idx])
        
        # Validation check
        feature_file_path = os.path.join(self.feature_folder, key + '.npy')
        if not os.path.exists(feature_file_path) or key not in self.captions:
            return None, None
        
        feats = load_feats(key, self.feature_folder)
        duration = self.captions[key]['duration']
        frames = feats.shape[0]

        captions = self.captions[key]['sentences']
        gt_timestamps = self.captions[key]['timestamps']
        gt_frames = (np.array(gt_timestamps) / duration * frames).astype(int)

        # Create [batch size, 64, feature dimension] batches, pad to make divisible by 32
        pad_len = (32 - frames % 32) % 32
        feats_padded = np.pad(feats, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
        frames_padded = feats_padded.shape[0]

        batch_size = (frames_padded // 32) + 1
        feature_batches = np.zeros((batch_size, 64, feats.shape[1]))
        caption_batches = [""] * batch_size

        for i in range(batch_size):
            start_idx = max(0, (i - 1) * 32)
            end_idx = min(frames_padded, (i + 1) * 32)

            if i == 0:
                feature_batches[i, 32:] = feats_padded[start_idx:end_idx]
            elif i == batch_size - 1:
                feature_batches[i, :end_idx - start_idx] = feats_padded[start_idx:end_idx]
            else:
                feature_batches[i, :32] = feats_padded[start_idx:start_idx + 32]
                feature_batches[i, 32:] = feats_padded[start_idx + 32:end_idx]

            # Assign captions to batches
            for caption, (start, end) in zip(captions, gt_frames):
                if start_idx <= start < end_idx or start_idx <= end < end_idx or (start < start_idx and end > end_idx):
                    if caption_batches[i] == "":
                        caption_batches[i] = caption
                    else:
                        caption_batches[i] += caption

        # # tokenize captions
        # tokenizer = GPT2Tokenizer.from_pretrained("./gpt2")
        # tokenizer.pad_token_id = tokenizer.eos_token_id
        # caption_batches = [f"{tokenizer.bos_token}{caption}{tokenizer.eos_token}" for caption in caption_batches]
        # caption_batches = [tokenizer.encode(caption) for caption in caption_batches]
        # max_len = max([len(caption) for caption in caption_batches])
        # caption_batches = [caption + [tokenizer.pad_token_id] * (max_len - len(caption)) for caption in caption_batches]

        return feature_batches, caption_batches


    def __len__(self):
        return len(self.all_feature_batches)

    def __getitem__(self, idx):
        feature_batch = torch.tensor(self.all_feature_batches[idx], dtype=torch.float32)
        caption_batch = self.all_caption_batches[idx]
        return feature_batch, caption_batch

def create_dataloaders(
        train_id_path, 
        train_caption_path, 
        test_id_path, 
        test_caption_path, 
        feature_folder, 
        batch_size=240, 
    ):
    # Load the data
    train_ids = load_json(train_id_path)
    train_captions = load_json(train_caption_path)
    test_ids = load_json(test_id_path)
    test_captions = load_json(test_caption_path)
    
    # Create datasets
    train_dataset = VideoCaptionDataset(train_ids, train_captions, feature_folder, batch_size)
    test_dataset = VideoCaptionDataset(test_ids, test_captions, feature_folder, batch_size)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=False)
    
    return train_loader, test_loader


if __name__ == '__main__':
    # Example usage
    dir = './data/activitynet-captions'
    train_id_path = os.path.join(dir, 'captions/train_ids.json')
    train_caption_path = os.path.join(dir, 'captions/train.json')
    test_id_path = os.path.join(dir, 'captions/val_ids.json')
    test_caption_path = os.path.join(dir, 'captions/val_1.json')
    feature_folder = os.path.join(dir, 'c3d')

    train_loader, test_loader = create_dataloaders(train_id_path, train_caption_path, test_id_path, test_caption_path, feature_folder, batch_size=1024)

    # Check for available device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loop through the training data
    for batch_features, batch_captions in train_loader:
        batch_features = batch_features.to(device)
        # batch_captions would need to be tokenized and converted to tensor based on the language model
        # Process your batch here
        print(batch_features.shape)  # Should be [1, 1024, 64, 500]
        # print(batch_captions)        # Corresponding captions