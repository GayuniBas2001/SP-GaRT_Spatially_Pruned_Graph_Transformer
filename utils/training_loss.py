import torch

def gravity_consistency_loss(predicted, observed,
                              root_idx=0,
                              lankle_idx=6,
                              rankle_idx=3,
                              margin=0.15):
    """
    Gravity-Consistency Loss for training.
    
    Penalises frames where the root joint (CoM proxy)
    projects outside the ankle-based base of support,
    but ONLY for standing frames (root Z above threshold).
    
    This is a SOFT, DIFFERENTIABLE version of GVR.
    Uses relu instead of boolean checks so gradients flow.
    
    H3.6M convention: Z is vertical. Floor = XY plane.
    
    Args:
        predicted:   (B, T_pred, J, 3)
        observed:    (B, T_obs,  J, 3)  — used to detect
                     whether the sequence is standing
        root_idx:    0  (root/pelvis)
        lankle_idx:  6  (left ankle)
        rankle_idx:  3  (right ankle)
        margin:      base of support margin in metres (0.15m)
    Returns:
        scalar differentiable loss
    """
    # ── 1. Detect standing frames from observed sequence ──────
    # Use mean root Z of observed frames as posture indicator
    # If root Z > threshold, person is standing
    STANDING_Z_THRESHOLD = 0.70  # metres

    obs_root_z = observed[:, :, root_idx, 2]    # (B, T_obs)
    mean_root_z = obs_root_z.mean(dim=1)         # (B,)
    is_standing = (mean_root_z > STANDING_Z_THRESHOLD)
                                                  # (B,) boolean

    # If no sequences in this batch are standing, return zero
    if is_standing.sum() == 0:
        return torch.tensor(0.0, requires_grad=True,
                            device=predicted.device)

    # Filter to standing sequences only
    pred_standing = predicted[is_standing]        # (B', T, J, 3)

    # ── 2. Extract relevant joints on floor plane (XY) ────────
    root_xy   = pred_standing[:, :, root_idx,   :2]  # (B', T, 2)
    lankle_xy = pred_standing[:, :, lankle_idx, :2]  # (B', T, 2)
    rankle_xy = pred_standing[:, :, rankle_idx, :2]  # (B', T, 2)

    # ── 3. Base of support bounding box ───────────────────────
    bos_min = torch.min(lankle_xy, rankle_xy) - margin  # (B', T, 2)
    bos_max = torch.max(lankle_xy, rankle_xy) + margin  # (B', T, 2)

    # ── 4. Soft violation — differentiable via relu ───────────
    # relu(x) = max(0, x)
    # If root is inside BoS, these are all <= 0 → relu gives 0
    # If root is outside BoS, the violated side gives > 0
    violation_left  = torch.relu(bos_min[..., 0] - root_xy[..., 0])
    violation_right = torch.relu(root_xy[..., 0] - bos_max[..., 0])
    violation_back  = torch.relu(bos_min[..., 1] - root_xy[..., 1])
    violation_front = torch.relu(root_xy[..., 1] - bos_max[..., 1])

    # Total violation per frame = sum of all directional violations
    total_violation = (violation_left + violation_right +
                       violation_back + violation_front)  # (B', T)

    return total_violation.mean()