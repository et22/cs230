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

from fix_models.transforms import BaseVideoTransform, BaseImageTransform

class ImageDataset(Dataset):
    def __init__(self, image_dir: str, image_names: List[str], targets: NDArray[np.float32], image_ext = "JPEG", transform=None):
        self.image_dir = image_dir
        self.image_names = image_names
        self.image_ext = image_ext
        self.transform = transform
        self.targets = targets

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        targets = torch.from_numpy(self.targets[idx]).squeeze().float()
        image_name = self.image_names[idx]
        image_path = join(self.image_dir, f"{image_name}.{self.image_ext}")
        image = Image.open(image_path)
        
        if self.transform:
            image = self.transform(image)

        return image, targets

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

class ImageSearchDataset(Dataset):
    def __init__(self, image_dir: str, transform=None):
        self.image_dir = image_dir
        image_extensions = ('.jpg', '.png', '.JPEG') 
        self.image_names = [f for f in os.listdir(image_dir) if f.endswith(image_extensions)]       
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = join(self.image_dir, image_name)
        image = Image.open(image_path)
        
        if self.transform:
            image = self.transform(image)

        return image, image_name

class VideoSearchDataset(Dataset):
    def __init__(self, video_dir: str, num_frames=5, transform=None):
        self.video_dir = video_dir
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv') 
        self.video_names = [f for f in os.listdir(video_dir) if f.endswith(video_extensions)]
        self.transform = transform
        self.num_frames = num_frames

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        video_path = join(self.video_dir, video_name)

        cap = cv2.VideoCapture(video_path)

        frames = []
        for i in range(self.num_frames):
            _, frame = cap.read()
            image = Image.fromarray(frame)

            if self.transform:
                image = self.transform(image)
            
            frames.append(image)

        cap.release()

        frames = torch.permute(torch.stack(frames), (1, 0, 2, 3))

        return frames, video_name

# construct datasets and dataloaders for a given session
def get_datasets_and_loaders(input_dir, session_id, modality, exp_var_threshold, stim_dur_ms, stim_size, win_size, stimulus_dir, batch_size, first_frame_only = False, blur_sigma=0, pos=(400,180)):
    with open(join(input_dir, f"{session_id}.pickle"), "rb") as f:
        model_input = pickle.load(f)

    train_idx = pd.read_csv(join(input_dir, f"{session_id}.csv"))["train"].to_numpy() == 1
    test_idx = np.logical_not(train_idx)

    include_idx = model_input['expvar'] > exp_var_threshold

    train_targets = model_input['rates'][:, include_idx][train_idx, :] * stim_dur_ms/1000
    test_targets = model_input['rates'][:, include_idx][test_idx, :] * stim_dur_ms/1000

    train_stim = model_input['images'][train_idx]
    test_stim= model_input['images'][test_idx]

    if modality == "video":
        video_transform = BaseVideoTransform(output_size=(stim_size, stim_size), recenter_window=(win_size, win_size), blur_sigma=blur_sigma, x_center=pos[0], y_center=pos[1])
        train_dataset = VideoDataset(video_dir=stimulus_dir, video_names=train_stim, targets=train_targets, transform=video_transform, first_frame_only=first_frame_only)
        test_dataset = VideoDataset(video_dir=stimulus_dir, video_names=test_stim, targets=test_targets, transform=video_transform, first_frame_only=first_frame_only)
    elif modality == "image":
        image_transform = BaseImageTransform(output_size=(stim_size, stim_size), recenter_window=(win_size, win_size), x_center=pos[0], y_center=pos[1])
        train_dataset = ImageDataset(image_dir=stimulus_dir, image_names=train_stim, targets=train_targets, transform=image_transform)
        test_dataset = ImageDataset(image_dir=stimulus_dir, image_names=test_stim, targets=test_targets, transform=image_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = 2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers = 2)

    return train_dataset, test_dataset, train_loader, test_loader

# construct datasets and dataloaders for a given session
def get_search_dataset_and_loader(dataset_dir, modality, stim_size, win_size, batch_size):
    if modality == "video":
        video_transform = BaseVideoTransform(output_size=(stim_size, stim_size), recenter_window=(win_size, win_size))
        dataset = VideoSearchDataset(video_dir=dataset_dir, transform=video_transform)
    elif modality == "image":
        image_transform = BaseImageTransform(output_size=(stim_size, stim_size), recenter_window=(win_size, win_size))
        dataset = ImageSearchDataset(image_dir=dataset_dir, transform=image_transform)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers = 2)

    return dataset, loader