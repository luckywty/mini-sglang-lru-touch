#!/usr/bin/env bash
export CUDA_HOME=/usr/local/cuda-13.2
export CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14
export NVCC_PREPEND_FLAGS=--compiler-bindir=/usr/bin/g++-14
export PATH=/home/wangwei/ai-infra/Promotion/benchmark-env/bin:$CUDA_HOME/bin:$PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false MAX_JOBS=2
export FLASHINFER_CUDA_ARCH_LIST=12.0 TORCH_CUDA_ARCH_LIST=12.0
export MINISGL_DISABLE_OVERLAP_SCHEDULING=0
