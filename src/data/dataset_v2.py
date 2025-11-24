"""
Improved dataset with location frequency features.
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter


class LocationDataset(Dataset):
    """Next-location prediction dataset with frequency features."""
    
    def __init__(self, data_path, loc_freq=None, compute_freq=False):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        assert isinstance(self.data, list) and len(self.data) > 0
        
        # Compute location frequencies if requested
        if compute_freq:
            all_locs = []
            for item in self.data:
                all_locs.extend(item['X'])
                all_locs.append(item['Y'])
            self.loc_freq = Counter(all_locs)
            total = sum(self.loc_freq.values())
            self.loc_freq = {k: v/total for k, v in self.loc_freq.items()}
        else:
            self.loc_freq = loc_freq if loc_freq is not None else {}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get location frequencies
        loc_freqs = torch.FloatTensor([self.loc_freq.get(loc, 0.0) for loc in item['X']])
        
        return {
            'loc_seq': torch.LongTensor(item['X']),
            'user_seq': torch.LongTensor(item['user_X']),
            'weekday_seq': torch.LongTensor(item['weekday_X']),
            'start_min_seq': torch.LongTensor(item['start_min_X']),
            'dur_seq': torch.FloatTensor(item['dur_X']),
            'diff_seq': torch.LongTensor(item['diff']),
            'loc_freq': loc_freqs,
            'target': torch.LongTensor([item['Y']]).squeeze(),
        }


def collate_fn(batch):
    """Collate function for batching."""
    loc_seqs = [item['loc_seq'] for item in batch]
    user_seqs = [item['user_seq'] for item in batch]
    weekday_seqs = [item['weekday_seq'] for item in batch]
    start_min_seqs = [item['start_min_seq'] for item in batch]
    dur_seqs = [item['dur_seq'] for item in batch]
    diff_seqs = [item['diff_seq'] for item in batch]
    loc_freqs = [item['loc_freq'] for item in batch]
    targets = torch.stack([item['target'] for item in batch])
    
    lengths = torch.LongTensor([len(seq) for seq in loc_seqs])
    
    loc_seqs_padded = pad_sequence(loc_seqs, batch_first=True, padding_value=0)
    user_seqs_padded = pad_sequence(user_seqs, batch_first=True, padding_value=0)
    weekday_seqs_padded = pad_sequence(weekday_seqs, batch_first=True, padding_value=0)
    start_min_seqs_padded = pad_sequence(start_min_seqs, batch_first=True, padding_value=0)
    dur_seqs_padded = pad_sequence(dur_seqs, batch_first=True, padding_value=0.0)
    diff_seqs_padded = pad_sequence(diff_seqs, batch_first=True, padding_value=0)
    loc_freqs_padded = pad_sequence(loc_freqs, batch_first=True, padding_value=0.0)
    
    max_len = loc_seqs_padded.size(1)
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    
    return {
        'loc_seq': loc_seqs_padded,
        'user_seq': user_seqs_padded,
        'weekday_seq': weekday_seqs_padded,
        'start_min_seq': start_min_seqs_padded,
        'dur_seq': dur_seqs_padded,
        'diff_seq': diff_seqs_padded,
        'loc_freq': loc_freqs_padded,
        'lengths': lengths,
        'mask': mask,
        'target': targets,
    }


def create_dataloaders(train_path, val_path, test_path, batch_size=128, num_workers=2):
    """Create train/val/test dataloaders with frequency features."""
    train_dataset = LocationDataset(train_path, compute_freq=True)
    loc_freq = train_dataset.loc_freq
    
    val_dataset = LocationDataset(val_path, loc_freq=loc_freq)
    test_dataset = LocationDataset(test_path, loc_freq=loc_freq)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
