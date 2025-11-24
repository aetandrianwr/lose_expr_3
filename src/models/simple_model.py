"""
Simpler, more robust architecture for next-location prediction.
Focus: Better generalization, less overfitting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """Simple scaled dot-product attention."""
    
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        return torch.matmul(attn, v)


class SimpleTransformerLayer(nn.Module):
    """Lightweight transformer layer."""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.attn = ScaledDotProductAttention(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Multi-head attention
        q = self.q_proj(self.norm1(x)).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(self.norm1(x)).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(self.norm1(x)).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        attn_out = self.attn(q, k, v, mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        attn_out = self.out_proj(attn_out)
        
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        
        return x


class LocationPredictor(nn.Module):
    """
    Simplified location predictor with better regularization.
    """
    
    def __init__(
        self,
        num_locations,
        num_users,
        d_model=96,
        num_layers=2,
        num_heads=4,
        dropout=0.3,
    ):
        super().__init__()
        
        # Embeddings with smaller dimensions
        self.loc_embed = nn.Embedding(num_locations + 1, d_model, padding_idx=0)
        self.user_embed = nn.Embedding(num_users + 1, d_model // 4, padding_idx=0)
        self.weekday_embed = nn.Embedding(8, d_model // 8, padding_idx=0)
        
        # Simple temporal features
        self.temporal_proj = nn.Linear(4, d_model // 4)  # start_min, duration, diff, freq
        
        # Input fusion
        input_dim = d_model + d_model // 4 + d_model // 8 + d_model // 4
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Positional encoding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, 200, d_model) * 0.02)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            SimpleTransformerLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_locations)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
    
    def forward(self, batch):
        # Embeddings
        loc_emb = self.loc_embed(batch['loc_seq'])
        user_emb = self.user_embed(batch['user_seq'])
        weekday_emb = self.weekday_embed(batch['weekday_seq'])
        
        # Temporal features (normalized)
        temporal = torch.stack([
            batch['start_min_seq'].float() / 1440.0,
            torch.log1p(batch['dur_seq']) / 10.0,
            batch['diff_seq'].float() / 7.0,
            batch.get('loc_freq', torch.zeros_like(batch['dur_seq']))
        ], dim=-1)
        temporal_emb = self.temporal_proj(temporal)
        
        # Combine
        x = torch.cat([loc_emb, user_emb, weekday_emb, temporal_emb], dim=-1)
        x = self.input_proj(x)
        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_embed[:, :seq_len, :]
        
        # Transformer
        mask = batch['mask']
        for layer in self.layers:
            x = layer(x, mask)
        
        # Pool: weighted average by recency
        lengths = batch['lengths']
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1).float()
        recency_weights = torch.exp(-0.1 * (lengths.unsqueeze(1).float() - positions - 1))
        recency_weights = recency_weights.masked_fill(~mask, 0)
        recency_weights = recency_weights / (recency_weights.sum(dim=1, keepdim=True) + 1e-8)
        x = (x * recency_weights.unsqueeze(-1)).sum(dim=1)
        
        # Classify
        x = self.norm(x)
        logits = self.classifier(x)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(num_locations, num_users, config):
    """Create simplified model."""
    model = LocationPredictor(
        num_locations=num_locations,
        num_users=num_users,
        d_model=config.get('d_model', 96),
        num_layers=config.get('num_layers', 2),
        num_heads=config.get('num_heads', 4),
        dropout=config.get('dropout', 0.3),
    )
    
    print(f"Model created with {model.count_parameters():,} parameters")
    return model
