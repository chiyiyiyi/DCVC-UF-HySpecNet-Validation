#!/usr/bin/env python3
"""Prepare same-sensor HSI scenes as grayscale PNG sequences for DCVC-UF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io
from PIL import Image


DATA_ROOT = Path("/home/user/4T_Storage/ZX/HSI")

TRAIN_SCENES = [
    DATA_ROOT / "California1" / f"California1_data_{index}.mat"
    for index in range(1, 13)
] + [
    DATA_ROOT / "Arizona" / f"Arizona_data_{index}.mat"
    for index in range(1, 5)
]

VAL_SCENES = [
    DATA_ROOT / "California1" / "California1_data_200.mat",
    DATA_ROOT / "California1" / "California1_data_201.mat",
    DATA_ROOT / "Arizona" / "Arizona_data_40.mat",
    DATA_ROOT / "Arizona" / "Arizona_data_41.mat",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/user/4T_Storage/Luohaoming/DCVC-UF-HSI/data"),
    )
    return parser.parse_args()


def load_cube(path: Path) -> np.ndarray:
    cube = scipy.io.loadmat(path)["Xim"]
    if cube.ndim != 3 or cube.shape[:2] != (256, 256):
        raise ValueError(f"Unexpected HSI shape for {path}: {cube.shape}")
    return cube.astype(np.float32)


def write_split(
    split: str,
    scenes: list[Path],
    output_root: Path,
    value_min: float,
    value_max: float,
) -> None:
    split_root = output_root / split
    split_root.mkdir(parents=True, exist_ok=True)

    band_count = 0
    frame_names = []
    sequence_entries = []
    for scene_path in scenes:
        cube = load_cube(scene_path)
        if band_count == 0:
            band_count = cube.shape[2]
        if cube.shape[2] != band_count:
            raise ValueError(
                f"Band count mismatch: {scene_path} has {cube.shape[2]}, "
                f"expected {band_count}"
            )

        sequence_name = scene_path.stem
        sequence_root = split_root / sequence_name
        sequence_root.mkdir(parents=True, exist_ok=True)
        scaled = np.clip(
            (cube - value_min) * (255.0 / (value_max - value_min)),
            0,
            255,
        ).astype(np.uint8)
        for band_index in range(band_count):
            filename = f"im{band_index + 1:05d}.png"
            Image.fromarray(scaled[:, :, band_index], mode="L").save(
                sequence_root / filename,
                format="PNG",
            )

        sequence_entries.append(
            {
                "path": sequence_name,
                "width": int(cube.shape[1]),
                "height": int(cube.shape[0]),
                "seq_length": int(band_count),
            }
        )

    frame_names = [f"im{band_index + 1:05d}.png" for band_index in range(band_count)]
    description = {
        "seqs": sequence_entries,
        "frames": frame_names,
        "preprocessing": {
            "source": "uint16 Xim",
            "value_min": value_min,
            "value_max": value_max,
            "output": "8-bit grayscale PNG",
        },
    }
    with (split_root / "description.json").open("w", encoding="utf-8") as stream:
        json.dump(description, stream, indent=2)

    print(
        f"{split}: sequences={len(sequence_entries)} bands={band_count} "
        f"frames={len(sequence_entries) * band_count}"
    )


def main() -> None:
    args = parse_args()
    all_scenes = TRAIN_SCENES + VAL_SCENES
    missing = [str(path) for path in all_scenes if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing scenes: {missing}")

    minima = []
    maxima = []
    for scene_path in all_scenes:
        cube = load_cube(scene_path)
        minima.append(float(cube.min()))
        maxima.append(float(cube.max()))
    value_min = min(minima)
    value_max = max(maxima)
    print(f"dataset range: {value_min}..{value_max}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_split("train", TRAIN_SCENES, args.output_root, value_min, value_max)
    write_split("val", VAL_SCENES, args.output_root, value_min, value_max)


if __name__ == "__main__":
    main()
