import torch


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

if __name__ == "__main__":
    # Example usage
    predicted = torch.rand(2, 10, 17, 3)  # (B, T, J, 3)
    target = torch.rand(2, 10, 17, 3)     # (B, T, J, 3)

    print("MPJPE per frame:", mpjpe(predicted, target))
    print("MPJPE at horizons:", mpjpe_at_horizons(predicted, target))