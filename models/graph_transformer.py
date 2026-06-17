"""
models/graph_transformer.py

M2 — Dense Graph Transformer for Human Motion Prediction

Extends M1 by introducing anatomical graph structure.
The skeleton adjacency matrix is used as a learnable
attention bias — connected joints attend to each other
more strongly.

Architecture adds to M1:
    + Spatial attention with anatomical adjacency bias
    + Temporal attention across frames
    + Factorised spatio-temporal processing

All hyperparameters identical to M1 for fair comparison.
"""

import torch
import torch.nn as nn
import math
import sys
import os

# ── Constants ────────────────────────────────────────────────
# Anatomical skeleton edges for 17-joint H3.6M
SKELETON_EDGES_17 = [
    (0,1),(1,2),(2,3),
    (0,4),(4,5),(5,6),
    (0,7),(7,8),(8,9),(9,10),
    (8,11),(11,12),(12,13),
    (8,14),(14,15),(15,16),
]

def build_adjacency_matrix(n_joints=17, edges=SKELETON_EDGES_17):
    """
    Build binary adjacency matrix from edge list.
    
    A[i][j] = 1 if joint i and joint j share a bone.
    Made symmetric — if hip connects to knee,
    knee also connects to hip.
    Also adds self-connections (diagonal = 1) so each
    joint always attends to itself.
    
    Returns:
        adj: (J, J) float tensor with 0s and 1s
    """
    adj = torch.zeros(n_joints, n_joints)
    
    for (i, j) in edges:
        adj[i, j] = 1.0
        adj[j, i] = 1.0  # symmetric
    
    # Self-connections — each joint attends to itself
    adj = adj + torch.eye(n_joints)
    adj = adj.clamp(0, 1)  # ensure no value exceeds 1
    
    return adj


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding — identical to M1."""
    
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SpatialAttentionBlock(nn.Module):
    """
    Attention over joints at a single time step.
    
    At each time step, every joint attends to every other
    joint. The anatomical adjacency matrix provides a
    learnable bias — connected joints attend more strongly.
    
    Why spatial attention first?
    The skeleton's spatial structure at each moment
    constrains what motion is possible next. Understanding
    "how is the body configured right now?" helps predict
    "where will each joint go next?"
    
    Args:
        d_model:  embedding dimension (256)
        n_heads:  attention heads (4)
        n_joints: number of joints (17)
        adj:      (J, J) adjacency matrix
        dropout:  dropout rate
    """
    
    def __init__(self, d_model, n_heads, n_joints, adj, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        
        # Standard multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Learnable adjacency bias — one scalar per joint pair
        # Initialised from anatomical structure
        # Positive value = attend more, 0 = no bias
        # Shape: (J, J) — same for all heads and all batches
        adj_bias = adj.clone()
        # Scale initial bias to be small
        adj_bias = adj_bias * 0.1
        self.adj_bias = nn.Parameter(adj_bias)
        
        # Feed-forward after attention
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Layer normalisation — stabilises training
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (B*T, J, d_model)
               Batch and time merged — process all time steps
               simultaneously as independent spatial snapshots
        Returns:
            x: (B*T, J, d_model) — same shape, enriched features
        """
        # ── Prepare adjacency bias ────────────────────────────
        # adj_bias shape: (J, J)
        # MultiheadAttention expects attn_bias: (J, J) or
        # (B*n_heads, J, J) — we use (J, J) which broadcasts
        # Clamp to prevent extreme values during training
        attn_bias = self.adj_bias.clamp(-10, 10)
        
        # ── Self-attention with anatomical bias ───────────────
        # Residual connection: output = input + attention(input)
        # This preserves the original information while adding
        # the attention-enriched version
        residual = x
        attn_out, _ = self.attention(
            x, x, x,
            attn_mask=attn_bias  # anatomical bias applied here
        )
        x = self.norm1(residual + self.dropout(attn_out))
        
        # ── Feed-forward ──────────────────────────────────────
        residual = x
        ff_out = self.feed_forward(x)
        x = self.norm2(residual + self.dropout(ff_out))
        
        return x


class TemporalAttentionBlock(nn.Module):
    """
    Attention over time steps for a single joint.
    
    For each joint independently, every time step attends
    to every other time step. No anatomical bias here —
    time relationships are learned freely.
    
    Why temporal attention after spatial?
    Once we know how the full body is configured at each
    moment (spatial), we model how each joint has been
    evolving over the observed sequence (temporal).
    
    Args:
        d_model: embedding dimension (256)
        n_heads: attention heads (4)
        t_obs:   number of time steps (10)
        dropout: dropout rate
    """
    
    def __init__(self, d_model, n_heads, t_obs=10, dropout=0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (B*J, T, d_model)
               Batch and joints merged — process all joints
               simultaneously as independent temporal sequences
        Returns:
            x: (B*J, T, d_model)
        """
        residual = x
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(residual + self.dropout(attn_out))
        
        residual = x
        ff_out = self.feed_forward(x)
        x = self.norm2(residual + self.dropout(ff_out))
        
        return x


class SpatioTemporalBlock(nn.Module):
    """
    One complete spatio-temporal processing block.
    
    Runs spatial attention then temporal attention.
    This order is chosen because:
    1. First understand the body configuration (spatial)
    2. Then understand how it evolves over time (temporal)
    
    This factorised approach processes 17×17 and 10×10
    matrices instead of 170×170, reducing computation
    while preserving both spatial and temporal dependencies.
    """
    
    def __init__(self, d_model, n_heads, n_joints,
                 t_obs, adj, dropout=0.1):
        super().__init__()
        self.spatial  = SpatialAttentionBlock(
            d_model, n_heads, n_joints, adj, dropout
        )
        self.temporal = TemporalAttentionBlock(
            d_model, n_heads, t_obs, dropout
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, T, J, d_model)
        Returns:
            x: (B, T, J, d_model)
        """
        B, T, J, d = x.shape
        
        # ── Spatial pass ──────────────────────────────────────
        # Merge batch and time: treat each (batch, time) pair
        # as an independent spatial snapshot of 17 joints
        x = x.reshape(B * T, J, d)
        x = self.spatial(x)
        x = x.reshape(B, T, J, d)
        
        # ── Temporal pass ─────────────────────────────────────
        # Merge batch and joints: treat each (batch, joint) pair
        # as an independent temporal sequence of 10 steps
        x = x.permute(0, 2, 1, 3)   # (B, J, T, d)
        x = x.reshape(B * J, T, d)
        x = self.temporal(x)
        x = x.reshape(B, J, T, d)
        x = x.permute(0, 2, 1, 3)   # back to (B, T, J, d)
        
        return x


class DenseGraphTransformer(nn.Module):
    """
    M2 — Dense Graph Transformer for Human Motion Prediction.
    
    Extends M1 (VanillaTransformer) by:
    1. Processing each joint as a separate node (not flattened)
    2. Running spatial attention with anatomical adjacency bias
    3. Running temporal attention over the observation window
    4. Factorised spatio-temporal blocks (space then time)
    
    All hyperparameters identical to M1 for fair comparison.
    The only architectural difference is the graph-guided
    spatio-temporal encoder replacing the vanilla encoder.
    
    Args:
        J:            joints (17)
        D:            spatial dims (3)
        d_model:      embedding dimension (256)
        n_heads:      attention heads (4)
        n_st_layers:  spatio-temporal block repetitions (2)
        d_ff:         feed-forward dim (512) — used in ST blocks
        dropout:      dropout rate (0.1)
        t_obs:        observed frames (10)
        t_pred:       predicted frames (25)
    """
    
    def __init__(self,
                 J=17, D=3,
                 d_model=256,
                 n_heads=4,
                 n_st_layers=2,
                 d_ff=512,
                 dropout=0.1,
                 t_obs=10,
                 t_pred=25):
        super().__init__()
        
        self.J      = J
        self.D      = D
        self.t_obs  = t_obs
        self.t_pred = t_pred
        
        # ── Build adjacency matrix ────────────────────────────
        adj = build_adjacency_matrix(J, SKELETON_EDGES_17)
        # Register as buffer — moves to GPU with model.to(device)
        # but is not a learned parameter
        self.register_buffer('adj', adj)
        
        # ── Joint embedding ───────────────────────────────────
        # Each joint gets its own learned embedding
        # This replaces M1's input projection of flattened joints
        # Now each joint is projected from 3D coords to d_model
        # independently — the model knows joints are separate
        self.joint_proj = nn.Linear(D, d_model)
        
        # ── Positional encoding over time ─────────────────────
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        
        # ── Spatio-temporal encoder blocks ───────────────────
        self.st_blocks = nn.ModuleList([
            SpatioTemporalBlock(
                d_model=d_model,
                n_heads=n_heads,
                n_joints=J,
                t_obs=t_obs,
                adj=adj,
                dropout=dropout
            )
            for _ in range(n_st_layers)
        ])
        
        # ── Pooling — from (B, T, J, d) to (B, T, d) ─────────
        # After spatio-temporal encoding, pool over joints
        # to get one vector per time step for the decoder
        # Simple mean pooling — averages information across joints
        # Alternative: learned attention pooling (more complex)
        # We use mean for simplicity and stability
        self.pool = lambda x: x.mean(dim=2)  # mean over joints
        
        # ── Decoder — identical to M1 ─────────────────────────
        # Using the same decoder as M1 for fair comparison
        # The only difference is what goes into the decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            dec_layer, num_layers=n_st_layers
        )
        
        # ── Query embeddings — identical to M1 ───────────────
        self.query_embed = nn.Embedding(t_pred, d_model)
        self.pos_enc_dec = PositionalEncoding(
            d_model, dropout=dropout
        )
        
        # ── Output projection ─────────────────────────────────
        # Project from d_model back to J*D joint space
        self.output_proj = nn.Linear(d_model, J * D)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, observed):
        """
        Args:
            observed: (B, T_obs, J, 3)
        Returns:
            predicted: (B, T_pred, J, 3)
        """
        B, T_obs, J, D = observed.shape
        
        # ── Step 1: Project each joint independently ──────────
        # (B, T, J, 3) → (B, T, J, 256)
        # Each joint gets its own 256-dim embedding
        # Unlike M1 which merged all joints into one vector
        x = self.joint_proj(observed)
        
        # ── Step 2: Add positional encoding over time ─────────
        # Apply to each joint's temporal sequence
        # Reshape to (B*J, T, d) for positional encoding
        x = x.permute(0, 2, 1, 3)         # (B, J, T, d)
        x = x.reshape(B * J, T_obs, -1)
        x = self.pos_enc(x)
        x = x.reshape(B, J, T_obs, -1)
        x = x.permute(0, 2, 1, 3)         # (B, T, J, d)
        
        # ── Step 3: Spatio-temporal encoding ──────────────────
        # Each block runs spatial attention then temporal attention
        for block in self.st_blocks:
            x = block(x)
        # x shape: (B, T_obs, J, 256)
        
        # ── Step 4: Pool over joints to get memory ────────────
        # (B, T_obs, J, 256) → (B, T_obs, 256)
        # Decoder needs one vector per time step
        memory = self.pool(x)
        
        # ── Step 5: Build decoder queries (same as M1) ────────
        query_idx = torch.arange(
            self.t_pred, device=observed.device
        )
        queries = self.query_embed(query_idx)
        queries = queries.unsqueeze(0).expand(B, -1, -1)
        queries = self.pos_enc_dec(queries)
        
        # ── Step 6: Decode (same as M1) ───────────────────────
        decoded = self.decoder(queries, memory)
        
        # ── Step 7: Project to joint space ────────────────────
        # (B, T_pred, 256) → (B, T_pred, 51)
        out = self.output_proj(decoded)
        
        # ── Step 8: Reshape ───────────────────────────────────
        # (B, T_pred, 51) → (B, T_pred, 17, 3)
        out = out.reshape(B, self.t_pred, J, D)
        
        return out
    
    def count_parameters(self):
        return sum(
            p.numel() for p in self.parameters()
            if p.requires_grad
        )