#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT=/home/user/4T_Storage/Luohaoming/DCVC-UF-HSI
SOURCE_ROOT=/home/user/4T_Storage/Luohaoming/DCVC-main
PYTHON_BIN=/home/user/4T_Storage/Luohaoming/env_dcvcuf/bin/python
ENV_LIB=/home/user/4T_Storage/Luohaoming/env_dcvcuf/lib
EPOCHS=${EPOCHS:-45}

cd "${SOURCE_ROOT}"
export CUDA_VISIBLE_DEVICES=0,1,2
export PYTHONPATH="${SOURCE_ROOT}:${EXPERIMENT_ROOT}/scripts"
export LD_LIBRARY_PATH="${ENV_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "${PYTHON_BIN}" "${EXPERIMENT_ROOT}/scripts/train_hyspecnet.py" \
    --model_structure hts \
    --model_path_i "${SOURCE_ROOT}/CVPR2026/cvpr2026_image.pth.tar" \
    --pretrain_path "${SOURCE_ROOT}/CVPR2026/cvpr2026_video_hts.pth.tar" \
    --train_dataset "${EXPERIMENT_ROOT}/data/hyspecnet/train" \
    --save_dir "${EXPERIMENT_ROOT}/runs/hyspecnet_hts_stage0" \
    --lambdas 1 768 \
    --training_scheduling stage0 \
    --batch_size 12 \
    -n 3 \
    -e "${EPOCHS}"
