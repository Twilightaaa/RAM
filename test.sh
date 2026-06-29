#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

NQ_FILE="${NQ_FILE:-/path/to/test.jsonl}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
RESTORE_FROM="${RESTORE_FROM:-/path/to/model}"
OUTPUT_DIR="${OUTPUT_DIR:-./eval_result}"
COMPRESSION_RATE="${COMPRESSION_RATE:-16}"
KEEP_RATIO="${KEEP_RATIO:-0.25}"
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29523}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
USE_TRANSFORM_LAYER="${USE_TRANSFORM_LAYER:-True}"
NUM_MEM_FUSION_LAYERS="${NUM_MEM_FUSION_LAYERS:-1}"

torchrun --master_port "${MASTER_PORT}" --nproc_per_node="${NUM_GPUS}" \
  ft_inference.py \
  --model_name_or_path "${MODEL_NAME}" \
  --test_file "${NQ_FILE}" \
  --restore_from "${RESTORE_FROM}" \
  --output_dir "${OUTPUT_DIR}" \
  --compression_rate "${COMPRESSION_RATE}" \
  --keep_ratio "${KEEP_RATIO}" \
  --num_samples "${NUM_SAMPLES}" \
  --per_device_eval_batch_size "${BATCH_SIZE}" \
  --bf16 True \
  --train False \
  --use_transform_layer "${USE_TRANSFORM_LAYER}" \
  --num_mem_fusion_layers "${NUM_MEM_FUSION_LAYERS}"
