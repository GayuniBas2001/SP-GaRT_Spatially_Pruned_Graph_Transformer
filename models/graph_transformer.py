"""
models/graph_transformer.py

M2 — Dense Graph Transformer for Human Motion Prediction

Extends M1 (VanillaTransformer) by introducing anatomical
graph structure into the encoder via a spatio-temporal
attention mechanism.

Key differences from M1:
    - Each joint is projected separately (not flattened)
    - Spatial attention uses learnable anatomical bias
    - Temporal attention processes each joint independently
    - Learned pooling preserves per-joint information
    - Decoder and loss are identical to M1 for fair comparison

All hyperparameters match M1 for valid ablation comparison.

Architecture:
    Input (B, T_obs, J, 3)
    → Joint projection    : (B, T_obs, J, 256)
    → Positional encoding : (B, T_obs, J, 256)
    → ST Block 1          : (B, T_obs, J, 256)
    → ST Block 2          : (B, T_obs, J, 256)
    → Learned pooling     : (B, T_obs, 256)
    → Decoder             : (B, T_pred, 256)
    → Output projection   : (B, T_pred, J*3)
    → Reshape             : (B, T_pred, J, 3)
"""

import math
import torch
import torch.nn as nn


# ── Skeleton definition ──────────────────────────────────────
# Must match SKELETON_EDGES_17 in data/h36m_dataset.py exactly
SKELETON_EDGES_17 = [
    (0, 1), (1, 2), (2, 3),           # root→rhip→rknee→rankle
    (0, 4), (4, 5), (5, 6),           # root→lhip→lknee→lankle
    (0, 7), (7, 8), (8, 9), (9, 10),  # root→spine→thorax→neck→head
    (8, 11), (11, 12), (12, 13),      # thorax→lshoulder→lelbow→lwrist
    (8, 14), (14, 15), (15, 16),      # thorax→rshoulder→relbow→rwrist
]


def build_adjacency_matrix(n_joints=17):
    """
    Build binary adjacency matrix from skeleton edge list.

    A[i,j] = 1 means joint i and joint j share a bone.
    Matrix is symmetric and includes self-connections.

    Returns:
        adj: (J, J) float tensor
    """
    adj = torch.zeros(n_joints, n_joints)
    for (i, j) in SKELETON_EDGES_17:
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    for i in range(n_joints):
        adj[i, i] = 1.0
    return adj


# ── Building blocks ──────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    Adds position-dependent signal so the model knows
    the order of frames in the sequence.
    """

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
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SpatialAttentionBlock(nn.Module):
    """
    Attention over the 17 joints at a single time step.

    Each joint attends to every other joint. Anatomically
    connected joints receive a learnable positive bias added
    to their attention score — they naturally attend more
    strongly to each other from the start of training.

    Uses manual attention implementation (not
    nn.MultiheadAttention) because the adjacency bias
    must be per-head and learnable — features not supported
    by nn.MultiheadAttention's attn_mask argument.

    Args:
        d_model : embedding dimension (256)
        n_heads : attention heads (4)
        adj     : (J, J) adjacency matrix
        dropout : dropout rate
    """

    def __init__(self, d_model, n_heads, adj, dropout=0.1):
        super().__init__()

        assert d_model % n_heads == 0, \
            "d_model must be divisible by n_heads"

        self.d_model  = d_model
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.scale    = self.head_dim ** -0.5

        # Q, K, V projections
        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Learnable adjacency bias: (n_heads, J, J)
        # Initialised: 0.1 for connected pairs, 0.0 otherwise
        # Training adjusts these — anatomy guides but
        # does not rigidly constrain
        n_joints  = adj.shape[0]
        bias_init = adj.unsqueeze(0).expand(
            n_heads, -1, -1
        ).clone() * 0.1
        self.adj_bias = nn.Parameter(bias_init)

        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.ff      = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B*T, J, d_model)
        Returns:
            x: (B*T, J, d_model)
        """
        BT, J, d = x.shape
        H  = self.n_heads
        hd = self.head_dim

        Q = self.q_proj(x).reshape(BT, J, H, hd).permute(0, 2, 1, 3)
        K = self.k_proj(x).reshape(BT, J, H, hd).permute(0, 2, 1, 3)
        V = self.v_proj(x).reshape(BT, J, H, hd).permute(0, 2, 1, 3)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        scores = scores + self.adj_bias.unsqueeze(0)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.permute(0, 2, 1, 3).reshape(BT, J, d)
        out = self.out_proj(out)

        x = self.norm1(x + self.dropout(out))
        x = self.norm2(x + self.ff(x))
        return x


class TemporalAttentionBlock(nn.Module):
    """
    Attention over the 10 time steps for each joint.

    For each joint independently, every time step attends
    to every other time step. No anatomical bias — time
    relationships are learned freely from data.

    Args:
        d_model : embedding dimension (256)
        n_heads : attention heads (4)
        dropout : dropout rate
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=1
        )

    def forward(self, x):
        """
        Args:
            x: (B*J, T, d_model)
        Returns:
            x: (B*J, T, d_model)
        """
        return self.encoder(x)


class SpatioTemporalBlock(nn.Module):
    """
    One complete spatio-temporal processing step.

    Spatial first: understand body configuration at each
    moment. Temporal second: understand how each joint
    evolves across observed frames.

    Designed to be stacked — M2 uses two of these blocks.

    Args:
        d_model : embedding dimension (256)
        n_heads : attention heads (4)
        adj     : (J, J) adjacency matrix
        dropout : dropout rate
    """

    def __init__(self, d_model, n_heads, adj, dropout=0.1):
        super().__init__()
        self.spatial  = SpatialAttentionBlock(
            d_model, n_heads, adj, dropout
        )
        self.temporal = TemporalAttentionBlock(
            d_model, n_heads, dropout
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, J, d_model)
        Returns:
            x: (B, T, J, d_model)
        """
        B, T, J, d = x.shape

        # Spatial: process all time steps simultaneously
        x = x.reshape(B * T, J, d)
        x = self.spatial(x)
        x = x.reshape(B, T, J, d)

        # Temporal: process all joints simultaneously
        x = x.permute(0, 2, 1, 3)     # (B, J, T, d)
        x = x.reshape(B * J, T, d)
        x = self.temporal(x)
        x = x.reshape(B, J, T, d)
        x = x.permute(0, 2, 1, 3)     # (B, T, J, d)

        return x


# ── Main model class ─────────────────────────────────────────

class DenseGraphTransformer(nn.Module):
    """
    M2 — Dense Graph Transformer for Human Motion Prediction.

    Extends M1 by introducing anatomical graph structure
    into the encoder. The skeleton adjacency matrix is used
    as a learnable attention bias in the spatial attention
    blocks — connected joints naturally attend more strongly
    to each other from the start of training.

    All hyperparameters match M1 for valid ablation.
    The only architectural difference is the encoder:
        M1: vanilla Transformer encoder on flattened joints
        M2: spatio-temporal encoder with graph-guided attention

    The decoder and loss function are identical to M1.

    Args:
        J           : joints (17)
        D           : spatial dims (3 for x,y,z)
        d_model     : embedding dimension (256)
        n_heads     : attention heads (4)
        n_st_layers : number of ST blocks stacked (2)
        d_ff        : decoder feed-forward dim (512)
        dropout     : dropout rate (0.1)
        t_obs       : observed frames (10)
        t_pred      : predicted frames (25)
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
        self.d_model = d_model
        self.t_obs  = t_obs
        self.t_pred = t_pred

        # Build adjacency matrix and register as buffer
        # register_buffer moves it to GPU with model.to(device)
        # but it is not a learned parameter
        adj = build_adjacency_matrix(J)
        self.register_buffer('adj', adj)

        # ── Encoder components ────────────────────────────────

        # Project each joint from 3D coords to d_model
        # independently — joints stay separate, not flattened
        self.joint_proj = nn.Linear(D, d_model)

        # Positional encoding over time steps
        self.pos_enc = PositionalEncoding(
            d_model, dropout=dropout
        )

        # Stack of spatio-temporal blocks
        self.st_blocks = nn.ModuleList([
            SpatioTemporalBlock(
                d_model, n_heads, adj, dropout
            )
            for _ in range(n_st_layers)
        ])

        # Learned pooling: (B, T, J, d) → (B, T, d)
        # Concatenates all joint vectors then projects
        # Preserves per-joint information unlike mean pooling
        self.pool_proj = nn.Linear(J * d_model, d_model)

        # ── Decoder components (identical to M1) ──────────────

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(
            dec_layer, num_layers=n_st_layers
        )

        # One learned query per future frame
        self.query_embed = nn.Embedding(t_pred, d_model)

        # Positional encoding for decoder queries
        self.pos_enc_dec = PositionalEncoding(
            d_model, dropout=dropout
        )

        # Project back to joint coordinate space
        self.output_proj = nn.Linear(d_model, J * D)

        self._init_weights()

    def _init_weights(self):
        """Xavier initialisation for all linear layers."""
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

        # ── Encoder ───────────────────────────────────────────

        # Project each joint independently
        # (B, T, J, 3) → (B, T, J, 256)
        x = self.joint_proj(observed)

        # Add positional encoding over time
        # Apply per-joint: merge batch+joint, encode, restore
        x = x.permute(0, 2, 1, 3)          # (B, J, T, d)
        x = x.reshape(B * J, T_obs, -1)
        x = self.pos_enc(x)
        x = x.reshape(B, J, T_obs, -1)
        x = x.permute(0, 2, 1, 3)          # (B, T, J, d)

        # Spatio-temporal encoding
        for block in self.st_blocks:
            x = block(x)
        # x: (B, T_obs, J, 256)

        # Learned pooling: joints → single summary per frame
        # (B, T, J, d) → (B, T, J*d) → (B, T, d)
        memory = self.pool_proj(
            x.reshape(B, T_obs, J * self.d_model)
        )
        # memory: (B, T_obs, 256)

        # ── Decoder (identical to M1) ─────────────────────────

        # Build future frame queries
        q_idx   = torch.arange(self.t_pred, device=observed.device)
        queries = self.query_embed(q_idx)
        queries = queries.unsqueeze(0).expand(B, -1, -1)
        queries = self.pos_enc_dec(queries)

        # Cross-attend to encoder memory
        decoded = self.decoder(queries, memory)

        # Project to joint space and reshape
        out = self.output_proj(decoded)
        out = out.reshape(B, self.t_pred, J, D)

        return out

    def count_parameters(self):
        """Return number of trainable parameters."""
        return sum(
            p.numel() for p in self.parameters()
            if p.requires_grad
        )