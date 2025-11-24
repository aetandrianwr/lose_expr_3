"""
Efficient Transformer-based next-location prediction model.
Designed for <500K params (Geolife) and <1M params (DIY).
NO RNN/LSTM - pure attention-based architecture with improved design.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """x: (batch_size, seq_len, d_model)"""
        return x + self.pe[:, :x.size(1), :]


class EfficientMultiHeadAttention(nn.Module):
    """Memory-efficient multi-head attention."""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.d_model)
        
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Efficient Transformer block with pre-norm."""
    
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        
        self.attn = EfficientMultiHeadAttention(d_model, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x


class NextLocationPredictor(nn.Module):
    """
    Improved Transformer for next-location prediction.
    
    Key improvements:
    - Better feature encoding with learned normalization
    - Richer temporal representations
    - Attention pooling instead of last-token pooling
    - Label smoothing capability
    """
    
    def __init__(
        self,
        num_locations,
        num_users,
        d_model=128,
        num_heads=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_len=200,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Enhanced embeddings with proper initialization
        self.loc_embedding = nn.Embedding(num_locations + 1, d_model, padding_idx=0)
        self.user_embedding = nn.Embedding(num_users + 1, d_model // 4, padding_idx=0)
        self.weekday_embedding = nn.Embedding(8, d_model // 8, padding_idx=0)
        
        # Improved temporal encoding
        self.start_min_proj = nn.Sequential(
            nn.Linear(1, d_model // 8),
            nn.LayerNorm(d_model // 8),
            nn.ReLU()
        )
        self.duration_proj = nn.Sequential(
            nn.Linear(1, d_model // 8),
            nn.LayerNorm(d_model // 8),
            nn.ReLU()
        )
        self.diff_proj = nn.Sequential(
            nn.Linear(1, d_model // 8),
            nn.LayerNorm(d_model // 8),
            nn.ReLU()
        )
        self.freq_proj = nn.Sequential(
            nn.Linear(1, d_model // 8),
            nn.LayerNorm(d_model // 8),
            nn.ReLU()
        )
        
        # Feature fusion
        fusion_dim = d_model + (d_model // 4) + 5 * (d_model // 8)
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Simpler attention pooling (parameter efficient)
        self.attention_pool = nn.Linear(d_model, 1)
        
        # Simpler output projection
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, num_locations)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly."""
        for name, p in self.named_parameters():
            if 'embedding' in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
    
    def forward(self, batch):
        # Extract and embed features
        loc_emb = self.loc_embedding(batch['loc_seq'])
        user_emb = self.user_embedding(batch['user_seq'])
        weekday_emb = self.weekday_embedding(batch['weekday_seq'])
        
        # Improved temporal features with normalization
        start_min_feat = self.start_min_proj(batch['start_min_seq'].unsqueeze(-1).float() / 1440.0)
        dur_feat = self.duration_proj(torch.log1p(batch['dur_seq'].unsqueeze(-1)) / 10.0)
        diff_feat = self.diff_proj(batch['diff_seq'].unsqueeze(-1).float() / 7.0)
        
        # Add frequency feature
        freq_feat = self.freq_proj(batch.get('loc_freq', torch.zeros_like(batch['dur_seq'])).unsqueeze(-1))
        
        # Concatenate all features
        combined = torch.cat([
            loc_emb, user_emb, weekday_emb,
            start_min_feat, dur_feat, diff_feat, freq_feat
        ], dim=-1)
        
        # Fuse features
        x = self.feature_fusion(combined)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer layers
        mask = batch['mask']
        for layer in self.transformer_layers:
            x = layer(x, mask)
        
        # Attention pooling (better than last-token)
        attn_weights = self.attention_pool(x)  # (B, L, 1)
        attn_weights = attn_weights.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=1)
        x = (x * attn_weights).sum(dim=1)  # (B, d_model)
        
        # Output projection
        x = self.output_norm(x)
        logits = self.output_proj(x)
        
        return logits
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(num_locations, num_users, config):
    """
    Factory function to create model.
    """
    model = NextLocationPredictor(
        num_locations=num_locations,
        num_users=num_users,
        d_model=config.get('d_model', 128),
        num_heads=config.get('num_heads', 4),
        num_layers=config.get('num_layers', 3),
        dim_feedforward=config.get('dim_feedforward', 256),
        dropout=config.get('dropout', 0.1),
        max_seq_len=config.get('max_seq_len', 200),
    )
    
    print(f"Model created with {model.count_parameters():,} parameters")
    return model
