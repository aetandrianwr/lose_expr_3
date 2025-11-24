"""
Dataset module for next-location prediction.
Implements PyTorch Dataset with proper batching and collation.
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class LocationDataset(Dataset):
    """Next-location prediction dataset."""
    
    def __init__(self, data_path):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to pickle file containing data
        """
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        # Verify data integrity
        assert isinstance(self.data, list), f"Expected list, got {type(self.data)}"
        assert len(self.data) > 0, "Empty dataset"
        
        # Verify first item structure
        sample = self.data[0]
        required_keys = ['X', 'user_X', 'weekday_X', 'start_min_X', 'dur_X', 'diff', 'Y']
        for key in required_keys:
            assert key in sample, f"Missing key: {key}"
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns:
            Dictionary with tensors
        """
        item = self.data[idx]
        
        return {
            'loc_seq': torch.LongTensor(item['X']),  # Location IDs
            'user_seq': torch.LongTensor(item['user_X']),  # User IDs
            'weekday_seq': torch.LongTensor(item['weekday_X']),  # Weekday
            'start_min_seq': torch.LongTensor(item['start_min_X']),  # Start minute
            'dur_seq': torch.FloatTensor(item['dur_X']),  # Duration
            'diff_seq': torch.LongTensor(item['diff']),  # Time diff
            'target': torch.LongTensor([item['Y']]).squeeze(),  # Target location
        }


def collate_fn(batch):
    """
    Collate function for batching variable-length sequences.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Dictionary of batched tensors with padding
    """
    # Separate components
    loc_seqs = [item['loc_seq'] for item in batch]
    user_seqs = [item['user_seq'] for item in batch]
    weekday_seqs = [item['weekday_seq'] for item in batch]
    start_min_seqs = [item['start_min_seq'] for item in batch]
    dur_seqs = [item['dur_seq'] for item in batch]
    diff_seqs = [item['diff_seq'] for item in batch]
    targets = torch.stack([item['target'] for item in batch])
    
    # Get sequence lengths
    lengths = torch.LongTensor([len(seq) for seq in loc_seqs])
    
    # Pad sequences (padding value = 0)
    loc_seqs_padded = pad_sequence(loc_seqs, batch_first=True, padding_value=0)
    user_seqs_padded = pad_sequence(user_seqs, batch_first=True, padding_value=0)
    weekday_seqs_padded = pad_sequence(weekday_seqs, batch_first=True, padding_value=0)
    start_min_seqs_padded = pad_sequence(start_min_seqs, batch_first=True, padding_value=0)
    dur_seqs_padded = pad_sequence(dur_seqs, batch_first=True, padding_value=0.0)
    diff_seqs_padded = pad_sequence(diff_seqs, batch_first=True, padding_value=0)
    
    # Create attention mask (1 for real tokens, 0 for padding)
    max_len = loc_seqs_padded.size(1)
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    
    return {
        'loc_seq': loc_seqs_padded,
        'user_seq': user_seqs_padded,
        'weekday_seq': weekday_seqs_padded,
        'start_min_seq': start_min_seqs_padded,
        'dur_seq': dur_seqs_padded,
        'diff_seq': diff_seqs_padded,
        'lengths': lengths,
        'mask': mask,
        'target': targets,
    }


def create_dataloaders(train_path, val_path, test_path, batch_size=128, num_workers=2):
    """
    Create train/val/test dataloaders.
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data
        test_path: Path to test data
        batch_size: Batch size
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = LocationDataset(train_path)
    val_dataset = LocationDataset(val_path)
    test_dataset = LocationDataset(test_path)
    
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
