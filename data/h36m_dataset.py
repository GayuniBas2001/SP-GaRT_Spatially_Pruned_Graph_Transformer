"""
h36m_dataset.py

Structured data loading, preprocessing, and visualization
for Human3.6M dataset for the SPaRTA research project.

Pipeline:
    .npz → unwrap → select 17 joints → downsample to 25Hz
    → sliding windows → PyTorch Dataset → DataLoader
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# ───────────────────────────────────────────────────────────────
# CONSTANTS
# ───────────────────────────────────────────────────────────────

# Indices of 17 standard joints within raw 32-joint H3.6M skeleton
H36M_JOINTS_17 = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15,
                   17, 18, 19, 25, 26, 27]

JOINT_NAMES_17 = [
    'root',      # 0
    'rhip',      # 1
    'rknee',     # 2
    'rankle',    # 3
    'lhip',      # 4
    'lknee',     # 5
    'lankle',    # 6
    'spine',     # 7
    'thorax',    # 8
    'neck',      # 9
    'head',      # 10
    'lshoulder', # 11
    'lelbow',    # 12
    'lwrist',    # 13
    'rshoulder', # 14
    'relbow',    # 15
    'rwrist'     # 16
]

# Anatomical edges (index into 17-joint list above)
SKELETON_EDGES_17 = [
    (0, 1), (1, 2), (2, 3),        # right leg
    (0, 4), (4, 5), (5, 6),        # left leg
    (0, 7), (7, 8), (8, 9),        # spine to neck
    (9, 10),                        # neck to head
    (8, 11), (11, 12), (12, 13),   # left arm
    (8, 14), (14, 15), (15, 16),   # right arm
]

# Standard train/test split by subject
TRAIN_SUBJECTS = ['S1', 'S5', 'S6', 'S7', 'S8']
TEST_SUBJECTS  = ['S9', 'S11']

# Evaluation horizons in milliseconds (at 25Hz, 1 frame = 40ms)
EVAL_HORIZONS_MS  = [80, 160, 320, 560, 1000]
EVAL_HORIZONS_IDX = [1,   3,   7,  13,   24]  # 0-indexed frame positions


# ───────────────────────────────────────────────────────────────
# DATASET CLASS
# ───────────────────────────────────────────────────────────────

class H36MDataset(Dataset):
    """
    PyTorch Dataset for Human3.6M 3D pose sequences.

    Loads raw 32-joint data, selects 17 standard joints,
    downsamples from 50Hz to 25Hz, and slices into
    overlapping (obs, future) windows.

    Args:
        npz_path:   Path to data_3d_h36m.npz
        subjects:   List of subject IDs e.g. ['S1', 'S5']
        t_obs:      Observation frames (default 10 = 400ms at 25Hz)
        t_pred:     Prediction frames  (default 25 = 1000ms at 25Hz)
        stride:     Step between windows.
                    Use stride=5 for training, stride=1 for testing.
        downsample: Temporal factor. 2 = 50Hz → 25Hz.

    Returns per item (all CPU tensors, move to device in training loop):
        observed: (t_obs, 17, 3)  float32
        future:   (t_pred, 17, 3) float32
        subject:  str
        action:   str
    """

    def __init__(self, npz_path, subjects,
                 t_obs=10, t_pred=25,
                 stride=1, downsample=2):

        self.t_obs   = t_obs
        self.t_pred  = t_pred
        self.seq_len = t_obs + t_pred

        raw = np.load(npz_path, allow_pickle=True)
        positions_3d = raw['positions_3d'].item()

        self.sequences = []
        self.metadata  = []

        for subject in subjects:
            if subject not in positions_3d:
                print(f"Warning: {subject} not in file, skipping.")
                continue

            for action, seq in positions_3d[subject].items():
                # (T_raw, 32, 3) → (T_raw, 17, 3)
                seq = seq[:, H36M_JOINTS_17, :]

                # (T_raw, 17, 3) → (T_25hz, 17, 3)
                seq = seq[::downsample]

                T = len(seq)
                if T < self.seq_len:
                    continue

                for start in range(0, T - self.seq_len + 1, stride):
                    self.sequences.append(seq[start:start + self.seq_len])
                    self.metadata.append((subject, action))

        print(f"[H36MDataset] {len(self.sequences)} windows | "
              f"subjects={subjects} | "
              f"t_obs={t_obs} t_pred={t_pred} | "
              f"stride={stride} | 25Hz")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        NOTE: Returns CPU tensors.
        In your training loop, always do:
            obs = batch['observed'].to(device)
            fut = batch['future'].to(device)
        """
        seq = self.sequences[idx]          # (seq_len, 17, 3)
        subject, action = self.metadata[idx]

        return {
            'observed': torch.FloatTensor(seq[:self.t_obs]),
            'future':   torch.FloatTensor(seq[self.t_obs:]),
            'subject':  subject,
            'action':   action,
        }


# ───────────────────────────────────────────────────────────────
# DATALOADER BUILDER
# ───────────────────────────────────────────────────────────────

def build_dataloaders(data_path,
                      batch_size=32,
                      t_obs=10,
                      t_pred=25,
                      train_stride=5,
                      test_stride=1):
    """
    Build train and test DataLoaders with correct settings.

    Args:
        data_path:    Path to data_3d_h36m.npz
        batch_size:   Batch size for both loaders
        t_obs:        Observation window length
        t_pred:       Prediction horizon length
        train_stride: Window stride for training (5 recommended)
        test_stride:  Window stride for testing  (1 for dense eval)

    Returns:
        train_loader, test_loader
    """
    train_dataset = H36MDataset(
        npz_path=data_path, subjects=TRAIN_SUBJECTS,
        t_obs=t_obs, t_pred=t_pred, stride=train_stride
    )
    test_dataset = H36MDataset(
        npz_path=data_path, subjects=TEST_SUBJECTS,
        t_obs=t_obs, t_pred=t_pred, stride=test_stride
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=True
    )

    return train_loader, test_loader


# ───────────────────────────────────────────────────────────────
# VISUALIZATION
# ───────────────────────────────────────────────────────────────

def plot_pose(ax, pose, color, title=None, elev=15, azim=70):
    """
    Render a single 17-joint skeleton onto a 3D matplotlib axis.

    Args:
        ax:    matplotlib 3D axis
        pose:  (17, 3) numpy array
        color: matplotlib color string
        title: optional axis title
    """
    ax.scatter(pose[:,0], pose[:,1], pose[:,2],
               c=color, s=20, zorder=3)

    for (i, j) in SKELETON_EDGES_17:
        ax.plot([pose[i,0], pose[j,0]],
                [pose[i,1], pose[j,1]],
                [pose[i,2], pose[j,2]],
                color=color, linewidth=1.5)

    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([0, 2])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.view_init(elev=elev, azim=azim)

    if title:
        ax.set_title(title, fontsize=9)


def visualize_sequence(obs, fut, save_path=None):
    """
    Visualize observed and future pose sequences side by side.

    Shows 3 evenly spaced observed frames (blue) and
    3 key future frames at 160ms, 560ms, 1000ms (red).

    Args:
        obs:       (t_obs, 17, 3) tensor or numpy array
        fut:       (t_pred, 17, 3) tensor or numpy array
        save_path: optional path to save figure
    """
    if torch.is_tensor(obs):
        obs = obs.numpy()
    if torch.is_tensor(fut):
        fut = fut.numpy()

    # Evenly spaced observed frames
    obs_indices = np.linspace(0, len(obs)-1, 3, dtype=int)

    # Key future frames: 160ms=frame3, 560ms=frame13, 1000ms=frame24
    fut_indices = [3, 13, 24]
    fut_labels  = ['160ms', '560ms', '1000ms']

    fig = plt.figure(figsize=(14, 5))

    for plot_i, frame_i in enumerate(obs_indices):
        ax = fig.add_subplot(2, 3, plot_i+1, projection='3d')
        plot_pose(ax, obs[frame_i], 'royalblue',
                  title=f"Obs t={frame_i} ({frame_i*40}ms)")

    for plot_i, (frame_i, label) in enumerate(
            zip(fut_indices, fut_labels)):
        ax = fig.add_subplot(2, 3, plot_i+4, projection='3d')
        plot_pose(ax, fut[frame_i], 'tomato',
                  title=f"Future {label}")

    plt.suptitle('Blue = Observed  |  Red = Future Ground Truth',
                 fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


# ───────────────────────────────────────────────────────────────
# QUICK TEST
# ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Adjust for your environment:
    # Colab: 'data/data_3d_h36m.npz'
    # Local: 'D:/L4S2/Research Project in AI/Research/iccv21_git_src/data_3d_h36m.npz'
    DATA_PATH = "D:/L4S2/Research Project in AI/Research/iccv21_git_src/data_3d_h36m.npz"

    train_loader, test_loader = build_dataloaders(
        DATA_PATH,
        batch_size=32,
        train_stride=5,
        test_stride=1
    )

    batch = next(iter(train_loader))

    print(f"observed shape: {batch['observed'].shape}")
    print(f"future shape:   {batch['future'].shape}")
    print(f"sample subject: {batch['subject'][0]}")
    print(f"sample action:  {batch['action'][0]}")

    # Expected:
    # observed shape: torch.Size([32, 10, 17, 3])
    # future shape:   torch.Size([32, 25, 17, 3])

    # Visualize one sample
    visualize_sequence(
        batch['observed'][0],
        batch['future'][0],
        save_path='skeleton_viz.png'
    )