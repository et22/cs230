import torch
import numpy as np
import cv2
import pickle
import pandas as pd
import os
from torch.utils.data import DataLoader

from torchvision.models.optical_flow import raft_large

from os.path import join
from PIL import Image
from torch.utils.data import Dataset
from typing import List
from numpy.typing import NDArray

from fix_models.transforms import BaseVideoTransform

class VideoDataset(Dataset):
    def __init__(self, video_dir: str, video_names: List[str], targets: NDArray[np.float32], video_ext = "mp4", num_frames=5, first_frame_only = False, transform=None, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.video_dir = video_dir
        self.video_names = video_names
        self.video_ext = video_ext
        self.transform = transform
        self.num_frames = num_frames
        self.targets = targets
        self.first_frame_only = first_frame_only

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        targets = torch.from_numpy(self.targets[idx]).squeeze().float()

        video_name = self.video_names[idx]
        video_path = join(self.video_dir, f"{video_name}.{self.video_ext}")

        cap = cv2.VideoCapture(video_path)

        frames = []
        for i in range(self.num_frames):
            _, frame = cap.read()
            image = Image.fromarray(frame)

            if self.transform:
                image = self.transform(image)

            if self.first_frame_only: 
                if i == 0:
                    image1 = image
            else:
                image1 = image
            frames.append(image1)

        cap.release()

        frames = torch.permute(torch.stack(frames), (1, 0, 2, 3))

        return frames, targets

# construct datasets and dataloaders for a given session
def get_datasets_and_loaders(input_dir, session_id, modality, exp_var_threshold, stim_dur_ms, stim_size, win_size, stimulus_dir, batch_size, first_frame_only = False, blur_sigma=0, pos=(400,180), test_bs = False):
    with open(join(input_dir, f"{session_id}.pickle"), "rb") as f:
        model_input = pickle.load(f)

    train_idx = pd.read_csv(join(input_dir, f"{session_id}.csv"))["train"].to_numpy() == 1
    test_idx = np.logical_not(train_idx)

    include_idx = model_input['expvar'] > exp_var_threshold

    train_targets = model_input['rates'][:, include_idx][train_idx, :] * stim_dur_ms/1000
    test_targets = model_input['rates'][:, include_idx][test_idx, :] * stim_dur_ms/1000

    train_stim = model_input['images'][train_idx]
    test_stim= model_input['images'][test_idx]

    video_transform = BaseVideoTransform(output_size=(stim_size, stim_size), recenter_window=(win_size, win_size), x_center=pos[0], y_center=pos[1])
    train_dataset = VideoDataset(video_dir=stimulus_dir, video_names=train_stim, targets=train_targets, transform=video_transform, first_frame_only=first_frame_only)
    test_dataset = VideoDataset(video_dir=stimulus_dir, video_names=test_stim, targets=test_targets, transform=video_transform, first_frame_only=first_frame_only)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = 2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers = 2)

    if test_bs:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = 2)

    return train_dataset, test_dataset, train_loader, test_loader