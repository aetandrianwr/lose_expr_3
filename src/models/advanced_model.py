"""
Advanced Next-Location Predictor with Transition-Aware Architecture.

Key innovations:
1. Explicit transition modeling (location pairs)
2. Multi-scale temporal attention
3. User-specific embeddings
4. Auxiliary transition loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TransitionAwareAttention(nn.Module):
    """Attention mechanism that explicitly models location transitions."""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Transition bias for modeling location pair patterns
        self.transition_bias = nn.Parameter(torch.zeros(1, num_heads, 1, 1))
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, x, mask=None):
        B, L, _ = x.size()
        
        q = self.q_proj(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Add transition-aware bias
        attn = attn + self.transition_bias
        
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        
        return self.out_proj(out)


class TransformerLayer(nn.Module):
    """Transformer layer with gated residual."""
    
    def __init__(self, d_model, num_heads, dim_ff, dropout=0.1):
        super().__init__()
        
        self.attn = TransitionAwareAttention(d_model, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Gating mechanism for adaptive residual
        self.gate = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        # Attention with gated residual
        attn_out = self.attn(self.norm1(x), mask)
        gate_val = torch.sigmoid(self.gate(x))
        x = x + gate_val * self.dropout(attn_out)
        
        # Feedforward
        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x


class AdvancedLocationPredictor(nn.Module):
    """
    State-of-the-art location predictor with:
    - Transition-aware modeling
    - Multi-head attention
    - Rich feature fusion
    - Auxiliary transition prediction
    """
    
    def __init__(
        self,
        num_locations,
        num_users,
        d_model=128,
        num_layers=3,
        num_heads=4,
        dim_ff=256,
        dropout=0.2,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_locations = num_locations
        
        # Embeddings
        self.loc_embed = nn.Embedding(num_locations + 1, d_model, padding_idx=0)
        self.user_embed = nn.Embedding(num_users + 1, d_model // 4, padding_idx=0)
        self.weekday_embed = nn.Embedding(8, d_model // 8, padding_idx=0)
        
        # Temporal encoding
        self.temporal_proj = nn.Sequential(
            nn.Linear(4, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU()
        )
        
        # Input projection
        input_dim = d_model + d_model // 4 + d_model // 8 + d_model // 4
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, 200, d_model) * 0.02)
        
        # Transformer encoder
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, dim_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Transition module (models last_loc -> target explicitly)
        self.transition_embed = nn.Embedding(num_locations + 1, d_model // 2, padding_idx=0)
        self.transition_proj = nn.Sequential(
            nn.Linear(d_model + d_model // 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Output heads
        self.norm = nn.LayerNorm(d_model)
        
        # Main classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_locations)
        )
        
        # Auxiliary transition classifier (helps regularization)
        self.transition_classifier = nn.Linear(d_model, num_locations)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
    
    def forward(self, batch, return_transition_logits=False):
        # Embeddings
        loc_emb = self.loc_embed(batch['loc_seq'])
        user_emb = self.user_embed(batch['user_seq'])
        weekday_emb = self.weekday_embed(batch['weekday_seq'])
        
        # Temporal features
        temporal = torch.stack([
            batch['start_min_seq'].float() / 1440.0,
            torch.log1p(batch['dur_seq']) / 10.0,
            batch['diff_seq'].float() / 7.0,
            batch.get('loc_freq', torch.zeros_like(batch['dur_seq']))
        ], dim=-1)
        temporal_emb = self.temporal_proj(temporal)
        
        # Combine features
        x = torch.cat([loc_emb, user_emb, weekday_emb, temporal_emb], dim=-1)
        x = self.input_proj(x)
        
        # Add positional encoding
        x = x + self.pos_embed[:, :x.size(1), :]
        
        # Transformer layers
        mask = batch['mask']
        for layer in self.layers:
            x = layer(x, mask)
        
        # Get last location representation
        lengths = batch['lengths']
        last_indices = (lengths - 1).clamp(min=0)
        batch_indices = torch.arange(x.size(0), device=x.device)
        last_repr = x[batch_indices, last_indices]
        
        # Get last location ID for transition modeling
        last_locs = batch['loc_seq'][batch_indices, last_indices]
        last_loc_emb = self.transition_embed(last_locs)
        
        # Combine sequence representation with transition
        combined = torch.cat([last_repr, last_loc_emb], dim=-1)
        combined = self.transition_proj(combined)
        
        # Final representation
        final_repr = self.norm(combined)
        
        # Main prediction
        logits = self.classifier(final_repr)
        
        if return_transition_logits:
            # Auxiliary transition prediction from last location
            transition_logits = self.transition_classifier(self.norm(last_repr))
            return logits, transition_logits
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(num_locations, num_users, config):
    """Create advanced model."""
    model = AdvancedLocationPredictor(
        num_locations=num_locations,
        num_users=num_users,
        d_model=config.get('d_model', 128),
        num_layers=config.get('num_layers', 3),
        num_heads=config.get('num_heads', 4),
        dim_ff=config.get('dim_ff', 256),
        dropout=config.get('dropout', 0.2),
    )
    
    print(f"Model created with {model.count_parameters():,} parameters")
    return model
