#!/usr/bin/env python3
"""Run the official DCVC-UF training loop on cached HySpecNet-11k patches."""

from __future__ import annotations

import sys

import train_video
from hyspecnet_video_dataset import HySpecNetFolder


_original_strategy = train_video.get_training_strategy


def hyspecnet_strategy(training_scheduling, frame_delay, model_structure):
    strategy = _original_strategy(training_scheduling, frame_delay, model_structure)
    return [
        [epoch, learning_rate, frames, cascaded, 128, 128]
        for epoch, learning_rate, frames, cascaded, _, _ in strategy
    ]


train_video.VideoFolder = HySpecNetFolder
train_video.get_training_strategy = hyspecnet_strategy


if __name__ == "__main__":
    train_video.main(sys.argv[1:])
