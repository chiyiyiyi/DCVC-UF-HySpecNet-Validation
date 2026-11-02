#!/usr/bin/env python3
"""Summarize official DCVC-UF test JSON into HSI bppc and quality statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.result.read_text(encoding="utf-8"))
    rows = []
    for dataset_name, sequences in data.items():
        for sequence_name, rates in sequences.items():
            for rate_name, result in rates.items():
                rows.append(
                    {
                        "dataset": dataset_name,
                        "sequence": sequence_name,
                        "rate": int(rate_name),
                        "bpppc": float(
                            result["ave_all_frame_bpp"]
                        ),
                        "psnr": float(result["ave_all_frame_psnr"]),
                    }
                )

    if not rows:
        raise RuntimeError("No bppc records found")

    for rate in sorted({row["rate"] for row in rows}):
        rate_rows = [row for row in rows if row["rate"] == rate]
        bpppc = np.asarray([row["bpppc"] for row in rate_rows], dtype=np.float64)
        psnr = np.asarray([row["psnr"] for row in rate_rows], dtype=np.float64)
        print(f"rate={rate} sequences={len(rate_rows)}")
        print(f"  bpppc_mean={bpppc.mean():.8f}")
        print(f"  bpppc_std={bpppc.std():.8f}")
        print(f"  psnr_mean={psnr.mean():.4f}")
        print(f"  psnr_std={psnr.std():.4f}")


if __name__ == "__main__":
    main()
