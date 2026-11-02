#!/usr/bin/env python3
"""Single-chunk synthetic smoke test for DCVC-UF HT-S on HSI-like grayscale bands."""

from __future__ import annotations

import argparse

import torch

from src.models.video_model_ht import DMC
from src.utils.common import ModelStructure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--bands", type=int, default=8)
    args = parser.parse_args()
    if args.bands != 8:
        raise ValueError("HT-S expects an 8-band chunk")

    torch.manual_seed(0)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    network = DMC(model_structure=ModelStructure.HTS).cuda().train()
    hsi = torch.rand(
        args.batch_size,
        args.bands,
        args.height,
        args.width,
        device="cuda",
    )
    zero_chroma = torch.zeros_like(hsi[:, :1])

    # DCVC-UF internally expects YUV-like 3-channel frames. A grayscale HSI band
    # is represented by its intensity in Y and zero-centered U/V channels.
    frames = []
    for band_index in range(args.bands):
        intensity = hsi[:, band_index : band_index + 1] - 0.5
        yuv = torch.cat((intensity, zero_chroma, zero_chroma), dim=1)
        frames.append(yuv)

    reference = frames[0]
    chunk = torch.cat(frames, dim=1)
    qp = torch.full((args.batch_size,), 31, dtype=torch.int32, device="cuda")
    lambdas = torch.ones((args.batch_size,), dtype=torch.float32, device="cuda")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    network.clear_dpb()
    network.add_ref_feature_from_frame(reference, apply_feature_adaptor=False)
    loss, info = network(
        chunk,
        qp,
        lambdas=lambdas,
        get_loss_info=True,
        curr_poc=1,
    )
    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), float("inf"))
    print(f"parameters={sum(p.numel() for p in network.parameters())}")
    print(f"loss={loss.item():.6f}")
    print(f"info={info}")
    print(f"grad_norm={grad_norm.item():.6f}")
    print(f"peak_memory_gb={torch.cuda.max_memory_allocated() / 1024**3:.2f}")
    print("SMOKE_TEST_OK")


if __name__ == "__main__":
    main()
