#!/usr/bin/env python3
"""Prepare byte-quantized HySpecNet test sequences for real-bitstream bppc."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/user/4T_Storage/Luohaoming/DCVC-UF-HSI/data/hyspecnet"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/home/user/4T_Storage/Luohaoming/DCVC-UF-HSI/data/hyspecnet_bppc"
        ),
    )
    parser.add_argument("--sequences", type=int, default=16)
    parser.add_argument("--bands", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_root = args.dataset_root / "test"
    files_file = test_root / "files.txt"
    if files_file.exists():
        files = [
            test_root / line.strip()
            for line in files_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        files = sorted(test_root.glob("*.npy"))
    if args.sequences < len(files):
        indices = np.linspace(
            0,
            len(files) - 1,
            args.sequences,
            dtype=np.int64,
        )
        files = [files[index] for index in indices]
    if not files:
        raise RuntimeError(f"No cached test patches found in {test_root}")

    sequence_root = args.output_root / "test"
    sequence_root.mkdir(parents=True, exist_ok=True)
    sequences = {}
    for source in files:
        cube = np.load(source)
        if cube.shape[0] < args.bands:
            raise ValueError(f"{source} has only {cube.shape[0]} bands")

        destination = sequence_root / source.stem
        destination.mkdir(parents=True, exist_ok=True)
        quantized = np.clip(
            np.rint(cube[: args.bands] * 255.0),
            0,
            255,
        ).astype(np.uint8)
        for band_index in range(args.bands):
            Image.fromarray(quantized[band_index], mode="L").save(
                destination / f"im{band_index + 1:05d}.png",
                format="PNG",
            )
        sequences[source.stem] = {
            "width": int(cube.shape[2]),
            "height": int(cube.shape[1]),
            "frames": args.bands,
            "intra_period": -1,
        }

    config = {
        "root_path": str(sequence_root),
        "test_classes": {
            "HySpecNet": {
                "test": 1,
                "base_path": "",
                "src_type": "png",
                "sequences": sequences,
            }
        },
    }
    with (args.output_root / "hyspecnet_bppc.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(config, stream, indent=2)

    print(
        f"prepared sequences={len(sequences)} bands={args.bands} "
        f"resolution={cube.shape[2]}x{cube.shape[1]}"
    )


if __name__ == "__main__":
    main()
