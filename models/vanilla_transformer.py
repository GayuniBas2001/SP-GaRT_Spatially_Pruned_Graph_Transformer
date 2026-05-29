"""
models/vanilla_transformer.py

M1 — Vanilla Transformer Baseline for Human Motion Prediction

Architecture:
    Observed sequence (B, T_obs, J, 3)
    → Flatten joints
    → Linear projection to d_model
    → Positional encoding
    → Transformer Encoder (2 layers, 4 heads)
    → Encoded memory
    → Learned query embeddings (one per future frame)
    → Positional encoding on queries
    → Transformer Decoder (2 layers, 4 heads)
    → Linear projection back to joint space
    → Reshape to (B, T_pred, J, 3)

No graph structure. No pruning. No physics loss.
This is the absolute baseline — M1 in the ablation study.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    Adds a position-dependent signal to each vector in the
    sequence so the model knows the order of frames.

    Without this, attention has no concept of which frame
    comes first — frame 5 and frame 1 would look identical
    to the model except for their content.
    """

    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Pre-compute the positional signals for all positions
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, d_model)
        # Add positional signal — does not change shape
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class VanillaTransformer(nn.Module):
    """
    Vanilla Seq2Seq Transformer for human motion prediction.

    This is M1 — the simplest possible baseline.
    No graph structure, no spatial pruning, no physics loss.
    Every subsequent model (M2, M3, M4) adds to this.

    Args:
        J:            number of joints (17 for Human3.6M)
        D:            spatial dimensions (3 for x,y,z)
        d_model:      internal embedding dimension (256)
        n_heads:      attention heads (4)
        n_enc_layers: encoder depth (2)
        n_dec_layers: decoder depth (2)
        d_ff:         feed-forward dimension (512)
        dropout:      dropout rate (0.1)
        t_obs:        observed frames (10)
        t_pred:       predicted frames (25)
    """

    def __init__(self,
                 J=17, D=3,
                 d_model=256,
                 n_heads=4,
                 n_enc_layers=2,
                 n_dec_layers=2,
                 d_ff=512,
                 dropout=0.1,
                 t_obs=10,
                 t_pred=25):
        super().__init__()

        self.J      = J
        self.D      = D
        self.t_pred = t_pred
        input_dim   = J * D  # 17 * 3 = 51

        # ── Input projection ────────────────────────────────
        # Converts the flattened 51-dim body description
        # to the model's working dimension of 256
        self.input_proj = nn.Linear(input_dim, d_model)

        # ── Positional encodings ─────────────────────────────
        # One for encoder (observed sequence)
        # One for decoder (future query sequence)
        self.pos_enc_encoder = PositionalEncoding(
            d_model, dropout=dropout
        )
        self.pos_enc_decoder = PositionalEncoding(
            d_model, dropout=dropout
        )

        # ── Transformer Encoder ──────────────────────────────
        # Reads the 10 observed frames and builds a compressed
        # understanding of the motion pattern
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True  # (B, T, d) — not (T, B, d)
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=n_enc_layers
        )

        # ── Transformer Decoder ──────────────────────────────
        # Generates 25 future frames by attending to
        # the encoded memory of the observed sequence
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            dec_layer,
            num_layers=n_dec_layers
        )

        # ── Learned query embeddings ─────────────────────────
        # One learnable vector per future frame
        # These are the "questions" the decoder asks:
        # "what will frame t+1 look like?", etc.
        self.query_embed = nn.Embedding(t_pred, d_model)

        # ── Output projection ────────────────────────────────
        # Converts 256-dim decoder output back to
        # 51-dim joint representation
        self.output_proj = nn.Linear(d_model, input_dim)

        # ── Weight initialisation ────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """
        Xavier initialisation for linear layers.
        Prevents vanishing/exploding gradients at start
        of training.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, observed):
        """
        Forward pass.

        Args:
            observed: (B, T_obs, J, 3) — observed skeleton sequence

        Returns:
            predicted: (B, T_pred, J, 3) — predicted future sequence
        """
        B, T_obs, J, D = observed.shape

        # ── Step 1: Flatten joints ───────────────────────────
        # (B, T_obs, J, 3) → (B, T_obs, 51)
        # Each time step is now one vector describing
        # the full body pose
        x = observed.reshape(B, T_obs, J * D)

        # ── Step 2: Project to model dimension ───────────────
        # (B, T_obs, 51) → (B, T_obs, 256)
        x = self.input_proj(x)

        # ── Step 3: Add positional encoding ──────────────────
        # Tells the encoder which frame is frame 1,
        # which is frame 2, etc.
        x = self.pos_enc_encoder(x)

        # ── Step 4: Encode ───────────────────────────────────
        # Each frame attends to all other frames
        # Output: compressed understanding of observed motion
        # Shape stays (B, T_obs, 256)
        memory = self.encoder(x)

        # ── Step 5: Build decoder queries ────────────────────
        # Create 25 learnable query vectors — one per future frame
        query_idx = torch.arange(
            self.t_pred, device=observed.device
        )
        queries = self.query_embed(query_idx)           # (T_pred, 256)
        queries = queries.unsqueeze(0).expand(B, -1, -1) # (B, T_pred, 256)

        # ── Step 6: Add positional encoding to queries ───────
        # Tells the decoder which query is for which time step
        queries = self.pos_enc_decoder(queries)

        # ── Step 7: Decode ───────────────────────────────────
        # Each query attends to the encoded memory
        # (cross-attention) to answer "what happens at time t?"
        # Shape: (B, T_pred, 256)
        decoded = self.decoder(queries, memory)

        # ── Step 8: Project back to joint space ──────────────
        # (B, T_pred, 256) → (B, T_pred, 51)
        out = self.output_proj(decoded)

        # ── Step 9: Reshape to skeleton format ───────────────
        # (B, T_pred, 51) → (B, T_pred, 17, 3)
        out = out.reshape(B, self.t_pred, J, D)

        return out

    def count_parameters(self):
        """Return number of trainable parameters."""
        return sum(
            p.numel() for p in self.parameters()
            if p.requires_grad
        )