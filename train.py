"""
Main training script for next-location prediction.
"""

import argparse
import yaml
import torch
import numpy as np
import random
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset_v2 import create_dataloaders
from src.models.advanced_model import create_model
from src.utils.trainer import Trainer


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train next-location prediction model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set seed
    set_seed(config['seed'])
    
    # Device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print(f"\n{'='*80}")
    print(f"Configuration: {config['dataset']['name'].upper()}")
    print(f"{'='*80}")
    for section, values in config.items():
        print(f"\n{section}:")
        if isinstance(values, dict):
            for k, v in values.items():
                print(f"  {k}: {v}")
        else:
            print(f"  {values}")
    print(f"{'='*80}\n")
    
    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_path=config['dataset']['train_path'],
        val_path=config['dataset']['val_path'],
        test_path=config['dataset']['test_path'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
    )
    
    print(f"✓ Train samples: {len(train_loader.dataset)}")
    print(f"✓ Val samples: {len(val_loader.dataset)}")
    print(f"✓ Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        num_locations=config['dataset']['num_locations'],
        num_users=config['dataset']['num_users'],
        config=config['model']
    )
    
    # Count parameters
    num_params = model.count_parameters()
    print(f"✓ Total parameters: {num_params:,}")
    
    # Check parameter constraint
    if config['dataset']['name'] == 'geolife':
        assert num_params < 500000, f"Geolife model must have <500K params, got {num_params:,}"
        print("✓ Parameter constraint satisfied (<500K for Geolife)")
    else:  # DIY
        assert num_params < 1000000, f"DIY model must have <1M params, got {num_params:,}"
        print("✓ Parameter constraint satisfied (<1M for DIY)")
    
    # Create trainer
    trainer_config = {
        **config['training'],
        'checkpoint_dir': config['training']['checkpoint_dir'],
        'num_classes': config['dataset']['num_locations']
    }
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=trainer_config,
        device=device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    test_perf = trainer.train(num_epochs=config['training']['num_epochs'])
    
    # Check final performance
    print(f"\n{'='*80}")
    print("PERFORMANCE CHECK")
    print(f"{'='*80}")
    
    if config['dataset']['name'] == 'geolife':
        target_acc = 40.0
        if test_perf['acc@1'] >= target_acc:
            print(f"✓ SUCCESS: Test Acc@1 {test_perf['acc@1']:.2f}% >= {target_acc}%")
        else:
            print(f"✗ FAILED: Test Acc@1 {test_perf['acc@1']:.2f}% < {target_acc}%")
    else:  # DIY
        target_acc = 45.0
        if test_perf['acc@1'] >= target_acc:
            print(f"✓ SUCCESS: Test Acc@1 {test_perf['acc@1']:.2f}% >= {target_acc}%")
        else:
            print(f"✗ FAILED: Test Acc@1 {test_perf['acc@1']:.2f}% < {target_acc}%")
    
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
