import torch

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

def gravity_violation_rate(predicted, lankle_idx=6, rankle_idx=3):
    """
    Gravity Violation Rate (GVR).

    Fraction of predicted frames where the CoM projection
    onto the floor plane (XZ) falls outside the base of support.

    Human3.6M coordinate system:
        X = left/right
        Y = vertical (up)       ← NOT the floor axis
        Z = forward/backward
    Floor plane = XZ (indices 0 and 2)

    Args:
        predicted:   (B, T, J, 3)
        lankle_idx:  index of left ankle in 17-joint list  (=6)
        rankle_idx:  index of right ankle in 17-joint list (=3)
    Returns:
        gvr: float in [0, 1]
    """
    # 1. Centre of mass
    com = centre_of_mass(predicted)          # (B, T, 3)

    # 2. Project onto floor = XZ plane (NOT XY)
    com_xz = torch.stack(
        [com[..., 0], com[..., 2]], dim=-1   # (B, T, 2)
    )

    # 3. Ankle positions on floor plane (XZ)
    lankle_xz = torch.stack([
        predicted[:, :, lankle_idx, 0],
        predicted[:, :, lankle_idx, 2]
    ], dim=-1)                               # (B, T, 2)

    rankle_xz = torch.stack([
        predicted[:, :, rankle_idx, 0],
        predicted[:, :, rankle_idx, 2]
    ], dim=-1)                               # (B, T, 2)

    # 4. Base of support bounding box
    bos_min = torch.min(lankle_xz, rankle_xz)   # (B, T, 2)
    bos_max = torch.max(lankle_xz, rankle_xz)   # (B, T, 2)

    # 5. Add margin (10cm = 0.10m) — realistic human stance width
    margin = 0.10
    bos_min = bos_min - margin
    bos_max = bos_max + margin

    # 6. Check if CoM is outside BoS
    outside = (
        (com_xz[..., 0] < bos_min[..., 0]) |
        (com_xz[..., 0] > bos_max[..., 0]) |
        (com_xz[..., 1] < bos_min[..., 1]) |
        (com_xz[..., 1] > bos_max[..., 1])
    )                                        # (B, T) boolean

    return outside.float().mean().item()

if __name__ == "__main__":
    # Example usage
    predicted = torch.rand(2, 10, 17, 3)  # (B, T, J, 3)
    target = torch.rand(2, 10, 17, 3)     # (B, T, J, 3)

    print("MPJPE per frame:", mpjpe(predicted, target))
    print("MPJPE at horizons:", mpjpe_at_horizons(predicted, target))