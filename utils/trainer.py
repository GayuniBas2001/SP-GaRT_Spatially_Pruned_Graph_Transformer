"""
trainer.py

Training loop, evaluation harness, inference speed measurement,
and results tracking for SPaRTA experiments.

Usage:
    from utils.trainer import train_model, evaluate_model,
                              measure_inference_time,
                              ResultsTracker
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from utils.metrics import (
    mpjpe_at_horizons, ade, fde,
    gravity_violation_rate, bone_length_error
)


# ───────────────────────────────────────────────────────────────
# EVALUATION HARNESS
# ───────────────────────────────────────────────────────────────

def evaluate_model(model, data_loader, device,
                   horizons_ms=[80, 160, 320, 560, 1000],
                   n_batches=None):
    """
    Full evaluation harness. Run this on ANY model.

    Every model trained in this project — M1 through M4 —
    passes through this function. Results are directly
    comparable because the evaluation protocol is identical.

    Args:
        model:        any model with forward(observed) → predicted
                      observed: (B, T_obs, J, 3)
                      predicted: (B, T_pred, J, 3)
        data_loader:  test DataLoader
        device:       torch device
        horizons_ms:  list of ms horizons to evaluate at
        n_batches:    if set, only evaluate on first n batches
                      use for quick checks during development
                      use None for full evaluation before reporting

    Returns:
        dict with keys: mpjpe, ade, fde, gvr, ble
    """
    model.eval()

    all_mpjpe = {ms: [] for ms in horizons_ms}
    all_ade   = []
    all_fde   = []
    all_gvr   = []
    all_ble   = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if n_batches and batch_idx >= n_batches:
                break

            obs  = batch['observed'].to(device)  # (B, 10, 17, 3)
            fut  = batch['future'].to(device)    # (B, 25, 17, 3)
            pred = model(obs)                    # (B, 25, 17, 3)

            # MPJPE at each horizon
            horizon_results = mpjpe_at_horizons(
                pred, fut, horizons_ms
            )
            for ms, val in horizon_results.items():
                all_mpjpe[ms].append(val)

            # ADE, FDE
            all_ade.append(ade(pred, fut).item())
            all_fde.append(fde(pred, fut).item())

            # GVR — returns None for seated batches, skip those
            gvr_val = gravity_violation_rate(pred)
            if gvr_val is not None:
                all_gvr.append(gvr_val)

            # BLE — needs observed for reference bone lengths
            # Move obs back to cpu for BLE (no grad needed)
            ble_val = bone_length_error(
                pred.cpu(), obs.cpu()
            )
            all_ble.append(ble_val)

    results = {}
    results['mpjpe'] = {
        ms: float(np.mean(vals))
        for ms, vals in all_mpjpe.items()
    }
    results['ade'] = float(np.mean(all_ade))
    results['fde'] = float(np.mean(all_fde))
    results['gvr'] = float(np.mean(all_gvr)) \
                     if all_gvr else None
    results['ble'] = float(np.mean(all_ble)) * 1000  # → mm

    return results


def print_results(results, model_name="Model"):
    """Pretty-print evaluation results for a single model."""
    print(f"\n{'='*52}")
    print(f"  {model_name}")
    print(f"{'='*52}")
    print(f"  MPJPE (mm) at horizons:")
    for ms, val in results['mpjpe'].items():
        print(f"    {ms:>6}ms : {val:>8.2f} mm")
    print(f"  ADE : {results['ade']:>8.2f} mm")
    print(f"  FDE : {results['fde']:>8.2f} mm")
    gvr_str = f"{results['gvr']:.4f}" \
              if results['gvr'] is not None else "N/A"
    print(f"  GVR : {gvr_str:>8}")
    print(f"  BLE : {results['ble']:>8.2f} mm")
    print(f"{'='*52}")


# ───────────────────────────────────────────────────────────────
# INFERENCE SPEED MEASUREMENT
# ───────────────────────────────────────────────────────────────

def measure_inference_time(model, device,
                            batch_size=1,
                            t_obs=10, J=17,
                            n_warmup=20, n_runs=200):
    """
    Measure mean inference latency in milliseconds.

    Uses batch_size=1 for real-time latency measurement.
    Warm-up runs ensure GPU is at steady state before timing.

    This is a core claim of the paper — the pruned model must
    be measurably faster than the dense graph baseline.
    Report mean ± std in the paper.

    Args:
        n_warmup: runs discarded before timing (GPU warm-up)
        n_runs:   number of timed runs to average over
    Returns:
        mean_ms: mean inference time in milliseconds
        std_ms:  standard deviation in milliseconds
    """
    model.eval()
    dummy = torch.randn(batch_size, t_obs, J, 3).to(device)

    # Warm up — GPU needs a few forward passes to reach
    # steady-state clock speed. Without this, first runs
    # are artificially slow.
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)

    # Synchronize GPU before starting timer
    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(dummy)
            # Synchronize ensures GPU finishes before
            # we stop the timer — critical for accuracy
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

    times = np.array(times)
    return float(times.mean()), float(times.std())


# ───────────────────────────────────────────────────────────────
# RESULTS TRACKER
# ───────────────────────────────────────────────────────────────

class ResultsTracker:
    """
    Tracks evaluation results for all models in one place.

    Use this throughout the project — add a row after every
    model is trained and evaluated.

    Usage:
        tracker = ResultsTracker()
        tracker.add('Zero-Velocity', zv_results, ms=0.12)
        tracker.add('M1_Vanilla',    m1_results, ms=4.32)
        tracker.print_table()
        tracker.save('results/results_table.csv')
    """

    def __init__(self):
        self.records = {}
        self.horizons = [80, 160, 320, 560, 1000]

    def add(self, model_name, eval_results, ms=None):
        """
        Add or update a model's results.

        Args:
            model_name:   string identifier for this model
            eval_results: dict returned by evaluate_model()
            ms:           inference latency in milliseconds
        """
        self.records[model_name] = {
            **eval_results,
            'inference_ms': ms
        }
        print(f"  Recorded results for: {model_name}")

    def print_table(self):
        """Print full comparison table to console."""
        h_headers = "  ".join(
            [f"{h}ms" for h in self.horizons]
        )
        header = (
            f"{'Model':<28}  {h_headers}"
            f"  {'ADE':>7}  {'FDE':>7}"
            f"  {'GVR':>7}  {'BLE':>7}  {'ms':>8}"
        )
        sep = "=" * len(header)
        print(f"\n{sep}")
        print(header)
        print(sep)

        for name, res in self.records.items():
            mpjpe_str = "  ".join([
                f"{res['mpjpe'].get(h, 0):>5.1f}"
                for h in self.horizons
            ])
            gvr_str = (
                f"{res['gvr']:>7.4f}"
                if res.get('gvr') is not None
                else f"{'N/A':>7}"
            )
            ms_str = (
                f"{res['inference_ms']:>8.2f}"
                if res.get('inference_ms') is not None
                else f"{'—':>8}"
            )
            print(
                f"{name:<28}  {mpjpe_str}"
                f"  {res.get('ade', 0):>7.1f}"
                f"  {res.get('fde', 0):>7.1f}"
                f"  {gvr_str}"
                f"  {res.get('ble', 0):>7.2f}"
                f"  {ms_str}"
            )
        print(sep)

    def save(self, path):
        """Save results table to CSV for later analysis."""
        import csv
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow(
                ['model'] +
                [f'mpjpe_{h}ms' for h in self.horizons] +
                ['ade', 'fde', 'gvr', 'ble_mm',
                 'inference_ms']
            )
            # Rows
            for name, res in self.records.items():
                row = [name]
                row += [
                    round(res['mpjpe'].get(h, 0), 2)
                    for h in self.horizons
                ]
                row += [
                    round(res.get('ade', 0), 2),
                    round(res.get('fde', 0), 2),
                    round(res['gvr'], 4)
                        if res.get('gvr') is not None
                        else 'N/A',
                    round(res.get('ble', 0), 2),
                    round(res['inference_ms'], 3)
                        if res.get('inference_ms') is not None
                        else 'N/A',
                ]
                writer.writerow(row)
        print(f"Results saved to: {path}")


# ───────────────────────────────────────────────────────────────
# TRAINING LOOP
# ───────────────────────────────────────────────────────────────

def train_one_epoch(model, train_loader, optimizer,
                    loss_fn, device):
    """Single training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in train_loader:
        obs = batch['observed'].to(device)
        fut = batch['future'].to(device)

        optimizer.zero_grad()
        pred = model(obs)
        loss = loss_fn(pred, fut)
        loss.backward()

        # Gradient clipping — prevents exploding gradients.
        # max_norm=1.0 is standard for transformer training.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / n_batches


def train_model(model, train_loader, test_loader,
                n_epochs, lr, device,
                experiment_name,
                model_config=None,
                loss_fn=None,
                eval_every=5,
                save_dir='checkpoints',
                log_dir='runs',
                resume=True):
    """
    Full training loop with logging, checkpointing, and resume.

    Use this for EVERY model — M1 through M4.
    Results are logged to TensorBoard and saved to Drive.

    Args:
        model:            PyTorch model
        train_loader:     training DataLoader
        test_loader:      test DataLoader
        n_epochs:         total epochs to train
        lr:               initial learning rate
        device:           torch device
        experiment_name:  string ID — used for filenames
        model_config:     dict of hyperparameters to save
                          in checkpoint (for reproducibility)
        loss_fn:          if None, uses MSELoss (L_recon)
        eval_every:       evaluate every N epochs
        save_dir:         where to save checkpoints
        log_dir:          where to save TensorBoard logs
        resume:           if True, auto-load latest checkpoint
    Returns:
        trained model
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f'{log_dir}/{experiment_name}', exist_ok=True)

    writer    = SummaryWriter(f'{log_dir}/{experiment_name}')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[30, 60, 80],
        gamma=0.5
    )
    loss_fn = loss_fn or nn.MSELoss()

    # ── Resume from checkpoint ────────────────────────────────
    start_epoch = 1
    best_mpjpe  = float('inf')
    latest_path = f'{save_dir}/{experiment_name}_latest.pth'
    best_path   = f'{save_dir}/{experiment_name}_best.pth'

    if resume and os.path.exists(latest_path):
        print(f"Resuming: {latest_path}")
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch'] + 1
        best_mpjpe  = ckpt.get('best_mpjpe', float('inf'))
        print(f"  Resumed at epoch {start_epoch} | "
              f"Best MPJPE@560ms: {best_mpjpe:.1f}mm")
    else:
        print(f"Starting: {experiment_name}")

    # ── Training loop ─────────────────────────────────────────
    for epoch in range(start_epoch, n_epochs + 1):

        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device
        )
        scheduler.step()

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar(
            'LR', optimizer.param_groups[0]['lr'], epoch
        )

        # Save latest checkpoint every epoch
        # This is what enables resume after Colab disconnect
        torch.save({
            'epoch':           epoch,
            'model_state':     model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_mpjpe':      best_mpjpe,
            'train_loss':      train_loss,
            'model_config':    model_config or {},
        }, latest_path)

        # Evaluate periodically
        if epoch % eval_every == 0:
            results   = evaluate_model(
                model, test_loader, device, n_batches=50
            )
            mpjpe_560 = results['mpjpe'][560]

            writer.add_scalar('MPJPE/560ms', mpjpe_560, epoch)
            if results['gvr'] is not None:
                writer.add_scalar('GVR', results['gvr'], epoch)

            print(
                f"Epoch {epoch:>3}/{n_epochs} | "
                f"loss: {train_loss:.5f} | "
                f"MPJPE@560ms: {mpjpe_560:.1f}mm | "
                f"GVR: {results['gvr'] or 'N/A'}"
            )

            # Save best checkpoint
            if mpjpe_560 < best_mpjpe:
                best_mpjpe = mpjpe_560
                torch.save({
                    'epoch':        epoch,
                    'model_state':  model.state_dict(),
                    'results':      results,
                    'best_mpjpe':   best_mpjpe,
                    'model_config': model_config or {},
                }, best_path)
                print(f"  ✓ Best saved: "
                      f"MPJPE@560ms={best_mpjpe:.1f}mm")

        elif epoch % 5 == 0:
            print(f"Epoch {epoch:>3}/{n_epochs} | "
                  f"loss: {train_loss:.5f}")

    writer.close()
    print(f"\nDone. Best MPJPE@560ms: {best_mpjpe:.1f}mm")
    print(f"Best checkpoint: {best_path}")
    return model