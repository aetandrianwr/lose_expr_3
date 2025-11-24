"""
Training engine with comprehensive monitoring and early stopping.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
import json
from pathlib import Path
from collections import Counter

from ..utils.metrics import calculate_correct_total_prediction, get_performance_dict


class Trainer:
    """Training orchestrator with validation and checkpointing."""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        config,
        device='cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        
        # Compute class weights (inverse frequency)
        print("Computing class weights...")
        all_targets = []
        for batch in train_loader:
            all_targets.extend(batch['target'].numpy().tolist())
        
        target_counts = Counter(all_targets)
        num_classes = config.get('num_classes', max(target_counts.keys()) + 1)
        
        # Compute inverse frequency weights
        class_weights = torch.ones(num_classes)
        total_samples = len(all_targets)
        for cls, count in target_counts.items():
            # Softer weighting: sqrt of inverse frequency
            class_weights[cls] = (total_samples / count) ** 0.5
        
        # Normalize weights
        class_weights = class_weights / class_weights.mean()
        
        print(f"Weight range: [{class_weights.min():.2f}, {class_weights.max():.2f}]")
        print(f"Most common class weight: {class_weights[max(target_counts, key=target_counts.get)]:.2f}")
        print(f"Least common class weight: {class_weights[min(target_counts, key=target_counts.get)]:.2f}")
        
        # Loss function with class weights
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device),
            label_smoothing=config.get('label_smoothing', 0.0)
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config.get('scheduler_patience', 5),
            verbose=True
        )
        
        # Tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.epochs_no_improve = 0
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        metrics_dict = {
            "correct@1": 0, "correct@3": 0, "correct@5": 0, "correct@10": 0,
            "rr": 0, "ndcg": 0, "f1": 0, "total": 0
        }
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            logits = self.model(batch)
            loss = self.criterion(logits, batch['target'])
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item() * batch['target'].size(0)
            
            with torch.no_grad():
                metrics, _, _ = calculate_correct_total_prediction(logits, batch['target'])
                metrics_dict["correct@1"] += metrics[0]
                metrics_dict["correct@3"] += metrics[1]
                metrics_dict["correct@5"] += metrics[2]
                metrics_dict["correct@10"] += metrics[3]
                metrics_dict["rr"] += metrics[4]
                metrics_dict["ndcg"] += metrics[5]
                metrics_dict["total"] += metrics[6]
            
            pbar.set_postfix({'loss': loss.item()})
        
        # Compute average metrics
        avg_loss = total_loss / metrics_dict["total"]
        perf = get_performance_dict(metrics_dict)
        
        return avg_loss, perf
    
    @torch.no_grad()
    def validate(self):
        """Validate on validation set."""
        self.model.eval()
        
        total_loss = 0
        metrics_dict = {
            "correct@1": 0, "correct@3": 0, "correct@5": 0, "correct@10": 0,
            "rr": 0, "ndcg": 0, "f1": 0, "total": 0
        }
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            logits = self.model(batch)
            loss = self.criterion(logits, batch['target'])
            
            # Metrics
            total_loss += loss.item() * batch['target'].size(0)
            
            metrics, _, _ = calculate_correct_total_prediction(logits, batch['target'])
            metrics_dict["correct@1"] += metrics[0]
            metrics_dict["correct@3"] += metrics[1]
            metrics_dict["correct@5"] += metrics[2]
            metrics_dict["correct@10"] += metrics[3]
            metrics_dict["rr"] += metrics[4]
            metrics_dict["ndcg"] += metrics[5]
            metrics_dict["total"] += metrics[6]
        
        # Compute average metrics
        avg_loss = total_loss / metrics_dict["total"]
        perf = get_performance_dict(metrics_dict)
        
        return avg_loss, perf
    
    @torch.no_grad()
    def test(self):
        """Test on test set."""
        self.model.eval()
        
        total_loss = 0
        metrics_dict = {
            "correct@1": 0, "correct@3": 0, "correct@5": 0, "correct@10": 0,
            "rr": 0, "ndcg": 0, "f1": 0, "total": 0
        }
        
        for batch in tqdm(self.test_loader, desc='Testing'):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            logits = self.model(batch)
            loss = self.criterion(logits, batch['target'])
            
            # Metrics
            total_loss += loss.item() * batch['target'].size(0)
            
            metrics, _, _ = calculate_correct_total_prediction(logits, batch['target'])
            metrics_dict["correct@1"] += metrics[0]
            metrics_dict["correct@3"] += metrics[1]
            metrics_dict["correct@5"] += metrics[2]
            metrics_dict["correct@10"] += metrics[3]
            metrics_dict["rr"] += metrics[4]
            metrics_dict["ndcg"] += metrics[5]
            metrics_dict["total"] += metrics[6]
        
        # Compute average metrics
        avg_loss = total_loss / metrics_dict["total"]
        perf = get_performance_dict(metrics_dict)
        
        return avg_loss, perf
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'config': self.config
        }
        
        # Save latest
        path = self.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, path)
        
        # Save best
        if is_best:
            path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, path)
            print(f"✓ Saved best model (val_loss={self.best_val_loss:.4f}, acc@1={self.best_val_acc:.2f}%)")
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        print(f"✓ Loaded checkpoint from {path}")
    
    def train(self, num_epochs):
        """Main training loop."""
        print(f"\n{'='*80}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"{'='*80}\n")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 80)
            
            # Train
            train_loss, train_perf = self.train_epoch()
            
            # Validate
            val_loss, val_perf = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_perf['acc@1'])
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_perf['acc@1'])
            
            # Print metrics
            print(f"\nTRAIN | Loss: {train_loss:.4f} | Acc@1: {train_perf['acc@1']:.2f}% | "
                  f"Acc@5: {train_perf['acc@5']:.2f}% | MRR: {train_perf['mrr']:.2f}% | "
                  f"NDCG: {train_perf['ndcg']:.2f}%")
            
            print(f"VAL   | Loss: {val_loss:.4f} | Acc@1: {val_perf['acc@1']:.2f}% | "
                  f"Acc@5: {val_perf['acc@5']:.2f}% | MRR: {val_perf['mrr']:.2f}% | "
                  f"NDCG: {val_perf['ndcg']:.2f}%")
            
            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_val_acc = val_perf['acc@1']
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.epochs_no_improve >= self.config.get('early_stopping_patience', 15):
                print(f"\n⚠ Early stopping triggered after {epoch} epochs (no improvement for {self.epochs_no_improve} epochs)")
                break
        
        print(f"\n{'='*80}")
        print("Training completed!")
        print(f"Best val_loss: {self.best_val_loss:.4f} | Best val_acc@1: {self.best_val_acc:.2f}%")
        print(f"{'='*80}\n")
        
        # Load best model and test
        print("Loading best model for final testing...")
        self.load_checkpoint(self.checkpoint_dir / 'best.pt')
        test_loss, test_perf = self.test()
        
        print(f"\n{'='*80}")
        print("FINAL TEST RESULTS")
        print(f"{'='*80}")
        print(f"Loss: {test_loss:.4f}")
        print(f"Acc@1:  {test_perf['acc@1']:.2f}%")
        print(f"Acc@5:  {test_perf['acc@5']:.2f}%")
        print(f"Acc@10: {test_perf['acc@10']:.2f}%")
        print(f"MRR:    {test_perf['mrr']:.2f}%")
        print(f"NDCG:   {test_perf['ndcg']:.2f}%")
        print(f"{'='*80}\n")
        
        # Save test results
        results = {
            'test_loss': test_loss,
            'test_metrics': test_perf,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        with open(self.checkpoint_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return test_perf
