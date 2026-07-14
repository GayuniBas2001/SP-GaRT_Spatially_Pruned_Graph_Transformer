"""
models/pruned_graph_transformer_v2.py

M3b — Learned Pruned Graph Transformer

Extends M2 by replacing the fixed heuristic pruning mask
(M3a: gaze + proximity rule) with a LEARNED relevance network
that discovers which joint pairs are most predictively relevant
from the training data itself.

Key difference from M3a:
    M3a: relevance = fixed_formula(gaze_score, proximity_score)
    M3b: relevance = LearnedRelevanceNetwork(positions, velocities)

The network learns context-appropriate attention weights:
- During reaching: arm chain joints become more relevant
- During walking: leg chain joints become more relevant
- Adapts automatically to activity context from data

Architecture identical to M2/M3a except:
    compute_pruning_mask() → LearnedRelevanceNetwork

No gravity loss — that remains M4's contribution.
Does NOT modify any existing model files.
"""

import torch
import torch.nn as nn

from models.graph_transformer import (
    build_adjacency_matrix,
    PositionalEncoding,
    TemporalAttentionBlock,
)
from models.pruned_graph_transformer import (
    PrunedSpatialAttentionBlock,
    PrunedSpatioTemporalBlock,
)


class LearnedRelevanceNetwork(nn.Module):
    """
    Learns joint-pair relevance from motion context.

    Instead of computing relevance from a hand-coded formula
    (gaze cone + proximity), this network learns WHICH joint
    relationships matter for prediction from training data.

    Input features per joint:
        - Last observed position  (3D)
        - Mean velocity across observation window (3D)
        → 6 features per joint

    Architecture:
        Joint encoder: 6 → 32 features per joint
        Pair scorer:   64 (concat of two joints) → 1 relevance score
        Output:        (B, J, J) matrix of relevance scores in [0,1]

    Why this is better than M3a's fixed rule:
        - Weights are learned, not hand-coded
        - Adapts to activity context automatically
        - Fast-moving joints get higher relevance than stationary ones
        - Discovers which kinematic chains matter for each motion type
    """

    def __init__(self, J=17, hidden_dim=32):
        super().__init__()
        self.J = J

        # Encode per-joint motion state
        # Input: position(3) + mean_velocity(3) = 6 features
        self.joint_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Score joint pairs from concatenated joint features
        # Input: features_i(hidden) + features_j(hidden)
        self.pair_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()   # output in [0, 1]
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, observed):
        """
        Args:
            observed: (B, T_obs, J, 3)
        Returns:
            relevance: (B, J, J) soft relevance scores in [0, 1]
                       High value = this joint pair should attend
                       more strongly to each other
        """
        B, T, J, _ = observed.shape

        # ── Per-joint motion features ─────────────────────────
        # Position: last observed frame
        last_pos = observed[:, -1, :, :]          # (B, J, 3)

        # Velocity: frame-to-frame difference averaged
        # over the observation window
        velocity  = observed[:, 1:, :, :] - \
                    observed[:, :-1, :, :]        # (B, T-1, J, 3)
        mean_vel  = velocity.mean(dim=1)           # (B, J, 3)

        # Concatenate position and velocity per joint
        joint_feat = torch.cat(
            [last_pos, mean_vel], dim=-1
        )                                          # (B, J, 6)

        # Encode each joint independently
        joint_enc = self.joint_encoder(joint_feat) # (B, J, hidden)

        # ── Pairwise relevance scoring ────────────────────────
        # For every pair (i, j), concatenate their encoded features
        # and pass through the pair scorer

        J_curr  = joint_enc.shape[1]
        feat_i  = joint_enc.unsqueeze(2).expand(
            -1, -1, J_curr, -1
        )                                          # (B, J, J, hidden)
        feat_j  = joint_enc.unsqueeze(1).expand(
            -1, J_curr, -1, -1
        )                                          # (B, J, J, hidden)
        pair_feat = torch.cat(
            [feat_i, feat_j], dim=-1
        )                                          # (B, J, J, hidden*2)

        # Score each pair → relevance in [0, 1]
        relevance = self.pair_scorer(pair_feat).squeeze(-1)
                                                   # (B, J, J)

        # Scale to attention bias range
        # Multiply by 0.5 so learned bias is comparable to
        # the anatomical bias initialised at 0.1
        relevance = relevance * 0.5

        return relevance


class LearnedPrunedGraphTransformer(nn.Module):
    """
    M3b — Learned Pruned Graph Transformer.

    Identical to M3a (PrunedGraphTransformer) except the
    heuristic pruning mask is replaced by LearnedRelevanceNetwork.

    Ablation purpose:
        M2  → M3a: does a fixed heuristic rule help?
        M2  → M3b: does a learned relevance network help?
        M3a → M3b: is learning better than hand-coding?

    The comparison M3a vs M3b directly tests the evaluator's
    concern: is a learnable, data-driven approach genuinely
    better than a rule-based one?
    """

    def __init__(self,
                 J=17, D=3,
                 d_model=256,
                 n_heads=4,
                 n_st_layers=2,
                 d_ff=512,
                 dropout=0.1,
                 t_obs=10,
                 t_pred=25,
                 relevance_hidden=32):
        super().__init__()

        self.J       = J
        self.D       = D
        self.d_model = d_model
        self.t_obs   = t_obs
        self.t_pred  = t_pred

        adj = build_adjacency_matrix(J)
        self.register_buffer('adj', adj)

        # ── Learned relevance network (replaces heuristic) ────
        self.relevance_net = LearnedRelevanceNetwork(
            J=J, hidden_dim=relevance_hidden
        )

        # ── Encoder — same structure as M2/M3a ────────────────
        self.joint_proj = nn.Linear(D, d_model)
        self.pos_enc    = PositionalEncoding(
            d_model, dropout=dropout
        )

        # Uses PrunedSpatioTemporalBlock from M3a —
        # the block accepts an optional pruning_mask argument
        self.st_blocks = nn.ModuleList([
            PrunedSpatioTemporalBlock(
                d_model, n_heads, adj, dropout
            )
            for _ in range(n_st_layers)
        ])

        self.pool_proj = nn.Linear(J * d_model, d_model)

        # ── Decoder — identical to M1, M2, M3a ───────────────
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.decoder     = nn.TransformerDecoder(
            dec_layer, num_layers=n_st_layers
        )
        self.query_embed = nn.Embedding(t_pred, d_model)
        self.pos_enc_dec = PositionalEncoding(
            d_model, dropout=dropout
        )
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

        # ── Compute LEARNED relevance mask ────────────────────
        # (B, J, J) — learned from joint positions + velocities
        relevance_mask = self.relevance_net(observed)

        # Expand for all time steps: (B, J, J) → (B*T, J, J)
        relevance_mask_BT = relevance_mask.unsqueeze(1).expand(
            -1, T_obs, -1, -1
        ).reshape(B * T_obs, J, J)

        # ── Encoder ───────────────────────────────────────────
        x = self.joint_proj(observed)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B * J, T_obs, -1)
        x = self.pos_enc(x)
        x = x.reshape(B, J, T_obs, -1)
        x = x.permute(0, 2, 1, 3)

        # ST blocks receive learned mask instead of heuristic mask
        for block in self.st_blocks:
            x = block(x, relevance_mask_BT)

        memory = self.pool_proj(
            x.reshape(B, T_obs, J * self.d_model)
        )

        # ── Decoder — identical to all other models ───────────
        q_idx   = torch.arange(
            self.t_pred, device=observed.device
        )
        queries = self.query_embed(q_idx)
        queries = queries.unsqueeze(0).expand(B, -1, -1)
        queries = self.pos_enc_dec(queries)

        decoded = self.decoder(queries, memory)
        out     = self.output_proj(decoded)
        out     = out.reshape(B, self.t_pred, J, D)

        return out

    def count_parameters(self):
        return sum(
            p.numel() for p in self.parameters()
            if p.requires_grad
        )