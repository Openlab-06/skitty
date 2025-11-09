#!/bin/bash
set -e  # 에러나면 즉시 종료

# GPU 0,1번만 사용
CUDA_VISIBLE_DEVICES=0,1 \
vllm serve Devocean-06/Spam_Filter-gemma \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-seqs 128 \
  --trust-remote-code