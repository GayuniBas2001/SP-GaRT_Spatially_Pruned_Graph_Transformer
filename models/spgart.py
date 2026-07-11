"""
models/sparta.py

M4 — SP-GaRT: Full Model
Dense Graph Transformer + Gravity-Consistency Loss

This is M2 (DenseGraphTransformer) with the gravity-consistency
loss integrated into the training objective.

Architecture is identical to M2.
The only difference is the training loss:
    M2: L_total = L_recon (MSE)
    M4: L_total = L_recon + lambda * L_gravity

No architectural changes — same encoder, same decoder.
"""

from models.graph_transformer import DenseGraphTransformer

# SP-GaRT IS the DenseGraphTransformer architecture
# The physics contribution is in the training loss, not the model
SPGaRT = DenseGraphTransformer