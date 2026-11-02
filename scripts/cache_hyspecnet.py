#!/usr/bin/env python3
"""Cache a representative subset of the official HySpecNet-11k hard splits."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import tifffile


DATASET_ROOT = Path(
    "/home/user/4T_Storage/Yuhang/dataset/Hyspecent_11k/"
    "hyspecnet-11k/hyspecnet-11k"
)
INVALID_CHANNELS = {
    126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
    136, 137, 138, 139, 140, 160, 161, 162, 163, 164, 165, 166,
}
VALID_CHANNELS = [index for index in range(224) if index not in INVALID_CHANNELS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/user/4T_Storage/Luohaoming/DCVC-UF-HSI/data/hyspecnet"),
    )
    parser.add_argument("--train", type=int, default=512)
    parser.add_argument("--val", type=int, default=64)
    parser.add_argument("--test", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def read_split(split: str) -> list[str]:
    csv_path = DATASET_ROOT / "splits" / "hard" / f"{split}.csv"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        return [row[0] for row in csv.reader(stream) if row]


def select_rows(rows: list[str], count: int) -> list[str]:
    if count >= len(rows):
        return rows
    indices = np.linspace(0, len(rows) - 1, count, dtype=np.int64)
    return [rows[index] for index in indices]


def convert_one(item: tuple[str, Path]) -> tuple[str, str, float]:
    relative_path, destination = item
    source = DATASET_ROOT / "patches" / relative_path
    cube = tifffile.imread(source)[VALID_CHANNELS]
    cube = np.clip(cube, 0, 10000).astype(np.float32) / 10000.0
    np.save(destination, cube)
    return source.name, str(destination), float(cube.max())


def cache_split(
    split: str,
    count: int,
    output_root: Path,
    workers: int,
) -> None:
    selected = select_rows(read_split(split), count)
    split_root = output_root / split
    split_root.mkdir(parents=True, exist_ok=True)

    jobs = []
    for relative_path in selected:
        sample_id = Path(relative_path).parent.name
        jobs.append((relative_path, split_root / f"{sample_id}.npy"))

    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(convert_one, job) for job in jobs]
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 32 == 0 or completed == len(futures):
                print(f"{split}: {completed}/{len(futures)}")

    with (split_root / "files.txt").open("w", encoding="utf-8") as stream:
        for path in sorted(split_root.glob("*.npy")):
            stream.write(path.name + "\n")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for split, count in (("train", args.train), ("val", args.val), ("test", args.test)):
        cache_split(split, count, args.output_root, args.workers)


if __name__ == "__main__":
    main()
