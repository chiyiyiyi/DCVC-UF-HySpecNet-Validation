#!/usr/bin/env python3
"""DCVC-UF-compatible dataset for cached HySpecNet-11k HSI patches."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class HySpecNetFolder(Dataset):
    def __init__(
        self,
        root_folder_path: str,
        patch_h: int,
        patch_w: int,
        qp_num: int,
        lambdas,
        frame_num: int = 9,
        group_of_pictures: int = 8,
    ) -> None:
        self.root = Path(root_folder_path)
        files_file = self.root / "files.txt"
        if files_file.exists():
            self.files = [
                self.root / line.strip()
                for line in files_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            self.files = sorted(self.root.glob("*.npy"))
        if not self.files:
            raise ValueError(f"No cached HySpecNet samples found in {self.root}")

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.qp_num = qp_num
        self.lambdas = lambdas
        self.frame_num = frame_num
        self.group_of_pictures = group_of_pictures
        self.band_limit = 200

    def __getitem__(self, index: int):
        cube = np.load(self.files[index], mmap_mode="r")
        if cube.ndim != 3 or cube.shape[1:] != (128, 128):
            raise ValueError(f"Unexpected cached shape: {cube.shape}")

        available_bands = min(self.band_limit, cube.shape[0])
        if self.frame_num > available_bands:
            raise ValueError(
                f"Requested {self.frame_num} frames, only {available_bands} bands"
            )

        start = random.randint(0, available_bands - self.frame_num)
        selected = np.asarray(
            cube[start : start + self.frame_num, : self.patch_h, : self.patch_w],
            dtype=np.float32,
        )
        if selected.shape[1:] != (self.patch_h, self.patch_w):
            raise ValueError(
                f"Requested patch {self.patch_h}x{self.patch_w}, "
                f"available shape is {selected.shape[1:]}"
            )

        zero_chroma = np.zeros(
            (1, self.patch_h, self.patch_w),
            dtype=np.float32,
        )
        frames = []
        for band in selected:
            intensity = band[None, :, :] - 0.5
            frames.append(
                torch.from_numpy(
                    np.concatenate((intensity, zero_chroma, zero_chroma), axis=0)
                )
            )

        video_data = [frames[0]]
        current_group = []
        for frame in frames[1:]:
            current_group.append(frame)
            if len(current_group) == self.group_of_pictures:
                video_data.append(torch.cat(current_group, dim=0))
                current_group = []
        if current_group:
            raise ValueError("Frame count must be one reference plus complete chunks")

        current_qp = random.randint(0, self.qp_num - 1)
        video_data.append(torch.tensor(current_qp, dtype=torch.int32))
        video_data.append(torch.tensor(self.lambdas[current_qp], dtype=torch.float32))
        return video_data

    def __len__(self) -> int:
        return len(self.files)

    def set_frame_num(self, frame_num: int) -> None:
        self.frame_num = frame_num

    def set_patch_size(self, patch_width: int, patch_height: int) -> None:
        self.patch_w = patch_width
        self.patch_h = patch_height
