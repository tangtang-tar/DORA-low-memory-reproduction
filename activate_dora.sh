#!/usr/bin/env bash

# DORA 项目的唯一根目录。后续环境、模型、数据和缓存均从这里展开。
export DORA_ROOT=/media/tangtang/Data/DORA

# 将各工具默认写入用户主目录的缓存改到 DORA 目录。
export CONDA_PKGS_DIRS="$DORA_ROOT/cache/conda_pkgs"
export CONDA_ENVS_PATH="$DORA_ROOT/envs"
export PIP_CACHE_DIR="$DORA_ROOT/cache/pip"
export HF_HOME="$DORA_ROOT/cache/huggingface"
export HF_HUB_CACHE="$DORA_ROOT/cache/huggingface/hub"
export HF_DATASETS_CACHE="$DORA_ROOT/cache/huggingface/datasets"
export TORCH_HOME="$DORA_ROOT/cache/torch"
export TRITON_CACHE_DIR="$DORA_ROOT/cache/triton"
export VLLM_CACHE_ROOT="$DORA_ROOT/cache/vllm"
export CUDA_CACHE_PATH="$DORA_ROOT/cache/cuda"
export XDG_CACHE_HOME="$DORA_ROOT/cache/xdg"
export XDG_CONFIG_HOME="$DORA_ROOT/cache/xdg-config"
export XDG_DATA_HOME="$DORA_ROOT/cache/xdg-data"
export CONDARC="$DORA_ROOT/configs/condarc"
export TMPDIR="$DORA_ROOT/tmp"

# 让 Python 能找到官方仓库 scripts/sal 包。
export PYTHONPATH="$DORA_ROOT/src/scripts${PYTHONPATH:+:$PYTHONPATH}"

# 激活指定路径中的 Conda 环境，不创建用户目录下的命名环境。
source /home/tangtang/miniconda3/etc/profile.d/conda.sh
conda activate "$DORA_ROOT/envs/dora"

cd "$DORA_ROOT/src"
