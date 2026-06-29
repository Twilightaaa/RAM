#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

NQ_FILE="${NQ_FILE:-/path/to/train.jsonl}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/ram_qwen3-4b}"

RESTORE_FROM="${RESTORE_FROM:-}"
COMPRESSION_RATE="${COMPRESSION_RATE:-16}"
KEEP_RATIO="${KEEP_RATIO:-0.25}"
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29522}"

DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-ds_config.json}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"

MAX_STEPS="${MAX_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-1}"
EVAL_STEPS="${EVAL_STEPS:-10000}"
LOGGING_STEPS="${LOGGING_STEPS:-100}"

COMPRESSION_RATE="${COMPRESSION_RATE:-16}"
KEEP_RATIO="${KEEP_RATIO:-0.25}"
USE_CONTRASTIVE_LOSS="${USE_CONTRASTIVE_LOSS:-True}"
CONTRASTIVE_LOSS_WEIGHT="${CONTRASTIVE_LOSS_WEIGHT:-1.0}"
USE_TRANSFORM_LAYER="${USE_TRANSFORM_LAYER:-True}"
NUM_MEM_FUSION_LAYERS="${NUM_MEM_FUSION_LAYERS:-1}"


torchrun --master_port "${MASTER_PORT}" --nproc_per_node="${NUM_GPUS}" \
  instruction_finetune.py \
  --model_name_or_path "${MODEL_NAME}" \
  --train_file "${NQ_FILE}" \
  --test_file "${NQ_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --train True \
  --bf16 True \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --per_device_train_batch_size "${BATCH_SIZE}" \
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --max_steps "${MAX_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --eval_steps "${EVAL_STEPS}" \
  --logging_steps "${LOGGING_STEPS}" \
  --compression_rate "${COMPRESSION_RATE}" \
  --keep_ratio "${KEEP_RATIO}" \
  --use_contrastive_loss "${USE_CONTRASTIVE_LOSS}" \
  --contrastive_loss_weight "${CONTRASTIVE_LOSS_WEIGHT}" \
  --use_transform_layer "${USE_TRANSFORM_LAYER}" \
  --num_mem_fusion_layers "${NUM_MEM_FUSION_LAYERS}" \
  # --restore_from "${RESTORE_FROM}" \

