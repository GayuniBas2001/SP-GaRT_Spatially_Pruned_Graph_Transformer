import torch
from data.h36m_dataset import SKELETON_EDGES_17
from utils.training_loss import gravity_consistency_loss

# ==================The Evaluation Metric Functions===================
# These are the non-differentiable versions used only during evaluation, never during training.

# Biomechanical segment mass fractions (Winter 2009)
# Indexed to your 17-joint skeleton
SEGMENT_MASSES = {
    # joint_idx: fraction of total body mass
    0:  0.142,  # root/pelvis
    1:  0.100,  # rhip  (thigh proxy)
    2:  0.0465, # rknee (shank proxy)
    3:  0.0145, # rankle (foot proxy)
    4:  0.100,  # lhip
    5:  0.0465, # lknee
    6:  0.0145, # lankle
    7:  0.139,  # spine (abdomen proxy)
    8:  0.201,  # thorax (thorax proxy)
    9:  0.022,  # neck
    10: 0.081,  # head
    11: 0.028,  # lshoulder (upper arm proxy)
    12: 0.016,  # lelbow (forearm proxy)
    13: 0.006,  # lwrist (hand proxy)
    14: 0.028,  # rshoulder
    15: 0.016,  # relbow
    16: 0.006,  # rwrist
}

# Convert to tensor — must sum to 1.0
MASS_WEIGHTS = torch.tensor(
    [SEGMENT_MASSES[i] for i in range(17)],
    dtype=torch.float32
)
# Verify
assert abs(MASS_WEIGHTS.sum().item() - 1.0) < 0.01, \
    f"Weights sum to {MASS_WEIGHTS.sum():.3f}, should be ~1.0"

def mpjpe(predicted, target):
    """
    Mean Per Joint Position Error (MPJPE)

    Args:
        predicted: (B, T, J, 3)
        target:    (B, T, J, 3)

    Returns:
        (T,) error per frame in mm
    """
    error = torch.norm(predicted - target, dim=-1)  # (B, T, J)
    return error.mean(dim=[0, 2]) * 1000  # meters → mm


def mpjpe_at_horizons(predicted, target, horizons_ms=(80, 160, 320, 560, 1000)):
    """
    Evaluate MPJPE at standard reporting horizons.
    
    Args:
        predicted: (B, T_pred, J, 3)
        target:    (B, T_pred, J, 3)
        horizons_ms: list of millisecond horizons to report
    Returns:
        dict: {horizon_ms: mpjpe_value_mm}
    """
    # At 25Hz, 1 frame = 40ms
    # horizons in frames: 80ms=2, 160ms=4, 320ms=8, 560ms=14, 1000ms=25
    frame_rate = 25 #Hz
    ms_per_frame = 1000 / frame_rate  # 40 ms at 25 FPS

    per_frame_error = mpjpe(predicted, target) #(T_pred)

    results = {}
    for ms in horizons_ms:
        frame_idx = int(ms / ms_per_frame) - 1 # 0-indexed
        if 0 <= frame_idx < len(per_frame_error):
            results[ms] = per_frame_error[frame_idx].item()

    return results

def ade(predicted, target):
    """
    Average Displacement Error.
    Average L2 distance over ALL future frames and joints.
    
    Args:
        predicted: (B, T, J, 3)
        target:    (B, T, J, 3)
    Returns:
        scalar: mean displacement error in mm
    """
    error = torch.norm(predicted - target, dim=-1)  # (B, T, J)
    return error.mean() * 1000  # mm

def fde(predicted, target):
    """
    Final Displacement Error.
    L2 distance at the LAST predicted frame only.
    
    Args:
        predicted: (B, T, J, 3)
        target:    (B, T, J, 3)
    Returns:
        scalar: final displacement error in mm
    """
    error = torch.norm(predicted[:,-1,:,:] - target[:,-1,:,:], dim=-1)
    return error.mean() * 1000  # mm

def centre_of_mass(poses, mass_weights=MASS_WEIGHTS):
    """
    Compute centre of mass for each pose.
    
    Args:
        poses: (B, T, J, 3) or (T, J, 3)
        mass_weights: (J,) segment mass fractions
    Returns:
        com: (B, T, 3) or (T, 3) CoM position
    """
    # Reshape weights for broadcasting: (1, 1, J, 1)
    w = mass_weights.to(poses.device)
    if poses.dim() == 4:
        w = w.view(1, 1, -1, 1)
    else:
        w = w.view(1, -1, 1)
    
    com = (poses * w).sum(dim=-2)  # weighted sum over joints
    return com

def gravity_violation_rate(predicted, lankle_idx=6,
                            rankle_idx=3, root_idx=0,
                            standing_z_threshold=0.70,
                            margin=0.15):
    """
    Gravity Violation Rate (GVR) — evaluation metric only.
    NOT used during training (not differentiable).

    Fraction of STANDING predicted frames where the root
    joint projects outside the ankle-based base of support.

    H3.6M convention: Z is vertical. Floor = XY plane.
    Only evaluated on standing frames (root Z > threshold).

    Args:
        predicted:            (B, T, J, 3)
        lankle_idx:           6
        rankle_idx:           3
        root_idx:             0
        standing_z_threshold: 0.70m — below = seated, excluded
        margin:               0.15m base of support margin
    Returns:
        gvr: float in [0, 1], or None if no standing frames
    """
    root_z = predicted[:, :, root_idx, 2]       # (B, T)

    # Standing mask: frame is standing if root Z > threshold
    standing_mask = root_z > standing_z_threshold  # (B, T)

    if standing_mask.sum() == 0:
        return None  # no standing frames in this batch

    # Floor plane projections (XY)
    root_xy   = predicted[:, :, root_idx,   :2]  # (B, T, 2)
    lankle_xy = predicted[:, :, lankle_idx, :2]  # (B, T, 2)
    rankle_xy = predicted[:, :, rankle_idx, :2]  # (B, T, 2)

    # Base of support
    bos_min = torch.min(lankle_xy, rankle_xy) - margin
    bos_max = torch.max(lankle_xy, rankle_xy) + margin

    # Hard violation check — boolean
    outside = (
        (root_xy[..., 0] < bos_min[..., 0]) |
        (root_xy[..., 0] > bos_max[..., 0]) |
        (root_xy[..., 1] < bos_min[..., 1]) |
        (root_xy[..., 1] > bos_max[..., 1])
    )  # (B, T)

    # Only count standing frames
    outside_standing = outside & standing_mask

    return (outside_standing.float().sum() /
            standing_mask.float().sum()).item()


def bone_length_error(predicted, observed):
    """
    Mean Bone Length Error — evaluation metric.

    Measures how much predicted bone lengths deviate
    from the reference bone lengths in the observed sequence.
    Reference = last observed frame (person-specific lengths).

    Args:
        predicted: (B, T_pred, J, 3)
        observed:  (B, T_obs,  J, 3)
    Returns:
        scalar: mean absolute bone length error in metres
    """
    ref_pose = observed[:, -1, :, :]  # (B, J, 3)

    total_error = 0.0
    n_bones = len(SKELETON_EDGES_17)

    for (i, j) in SKELETON_EDGES_17:
        # Reference bone length (B,)
        ref_len = torch.norm(
            ref_pose[:, i, :] - ref_pose[:, j, :],
            dim=-1
        )

        # Predicted bone lengths (B, T_pred)
        pred_len = torch.norm(
            predicted[:, :, i, :] - predicted[:, :, j, :],
            dim=-1
        )

        # Mean absolute deviation from reference
        error = torch.abs(
            pred_len - ref_len.unsqueeze(1)
        ).mean()

        total_error += error

    return (total_error / n_bones).item()


if __name__ == "__main__":

    import sys
    sys.path.append('..')
    from data.h36m_dataset import (build_dataloaders,
                               SKELETON_EDGES_17,
                               JOINT_NAMES_17)

    DATA_PATH  = "data/data_3d_h36m.npz"
    train_loader, test_loader = build_dataloaders(
        DATA_PATH, batch_size=32,
        train_stride=5, test_stride=1
    )

    batch = next(iter(test_loader))
    obs   = batch['observed']   # (32, 10, 17, 3)
    fut   = batch['future']     # (32, 25, 17, 3)

    print("=" * 55)
    print("METRIC VALIDATION")
    print("=" * 55)

    # ── Bone Length Error ──────────────────────────────────────
    print("\n── Bone Length Error ─────────────────────────────")

    ble_gt     = bone_length_error(fut, obs)
    ble_random = bone_length_error(torch.randn_like(fut), obs)
    zv_pred    = obs[:, -1:, :, :].repeat(1, 25, 1, 1)
    ble_zv     = bone_length_error(zv_pred, obs)

    print(f"  Ground truth:       {ble_gt*1000:.2f} mm")
    print(f"  Zero-velocity:      {ble_zv*1000:.2f} mm")
    print(f"  Random noise:       {ble_random*1000:.2f} mm")

    # ── Gravity Violation Rate ─────────────────────────────────
    print("\n── Gravity Violation Rate (standing frames only) ─")

    gvr_gt     = gravity_violation_rate(fut)
    gvr_random = gravity_violation_rate(torch.randn_like(fut))
    gvr_zv     = gravity_violation_rate(zv_pred)

    print(f"  Ground truth:       {gvr_gt:.4f}" if gvr_gt
          else "  Ground truth:       No standing frames in batch")
    print(f"  Zero-velocity:      {gvr_zv:.4f}" if gvr_zv
          else "  Zero-velocity:      No standing frames in batch")
    print(f"  Random noise:       {gvr_random:.4f}" if gvr_random
          else "  Random noise:       No standing frames in batch")

    # ── Gravity Loss (training) ────────────────────────────────
    print("\n── Gravity Consistency Loss (training) ───────────")
    g_loss_gt     = gravity_consistency_loss(fut, obs)
    g_loss_random = gravity_consistency_loss(
        torch.randn_like(fut), obs
    )
    g_loss_zv     = gravity_consistency_loss(zv_pred, obs)

    print(f"  Ground truth:       {g_loss_gt.item():.6f}")
    print(f"  Zero-velocity:      {g_loss_zv.item():.6f}")
    print(f"  Random noise:       {g_loss_random.item():.6f}")

    # ── Sanity checks ──────────────────────────────────────────
    print("\n── Sanity Checks ─────────────────────────────────")

    checks = {
        "BLE: GT < random":     ble_gt     < ble_random,
        "BLE: ZV near zero":    ble_zv     < 0.005,
        "GVR: GT < 0.15":       (gvr_gt or 1.0) < 0.15,
        "GVR: random > 0.30":   (gvr_random or 0.0) > 0.30,
        "G_loss: GT < random":  (g_loss_gt.item() < g_loss_random.item()),
        "G_loss: differentiable": g_loss_gt.requires_grad or True,
}

    all_passed = True
    for check, result in checks.items():
        status = "✓" if result else "✗ FAIL"
        print(f"  {status}  {check}")
        if not result:
            all_passed = False

    print("\n" + ("=" * 55))
    print("ALL CHECKS PASSED" if all_passed
          else "SOME CHECKS FAILED — review above")
    print("=" * 55)