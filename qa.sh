#!/bin/bash

# RAMArguments
batch_size=1 
distillation_temp=1.0
compressor_hidden_size=4096
num_compressor_encoder_layers=8
pool_window_size=4
min_num_documents=10
max_num_documents=10
instruction_name=empty
benchmark_metric=accuracy

checkpoint_path=/path-to-checkpoint/pytorch_model.bin
base_model="/path-to-model/Qwen3-4B-Instruct"
target_model="/path-to-model/Qwen3-4B-Instruct"

data_path=/path-to-dataset/test-NQ.jsonl

mem_size=128
output_dir="path-to-output-dir"
merge_size=8
mkdir -p ${output_dir}
report_site="wandb"
gradient_accumulation_steps=1
max_steps=100000
save_steps=10000
learning_rate=1e-4
num_mem_fusion_layers=1
is_post_append=False

is_train=True
fine_tune=True
is_random=False
prefix_type=rs_prefix

deepspeed_config=zero2.json

# add for tkdr
launch_tkdr=True
key_percentage=0.01
merge_sizes="2,4,8,16,32"
adaptive_pick=True
tau=0.99
lamda_select=True
lamda_merge=True

lora_the_encoder=False

# add for ablation study
coarse_grained_on=False
fine_grained_on=True
redun_coarse=True
redun_fine=False

# add for autoregressive modeling
is_split=True
is_autoregressive=False
segment_size=50

# we should set max_doc_tokens for rope scaling
max_doc_tokens=32768

# draw settings
draw=False

# parameter for encore compression
enable_encore=True
top_p=0.8
uniform_distribution=False
# top_p_list="0.5,0.25,0.125,0.0625,0.03125"
top_p_list="0.25,0.125"

# parameter for encore ablation study
use_only_org_tokens=False
use_mean_compressed_tokens=False
use_all_compress=False

master_port=12345
CUDA_VISIBLE_DEVICES="5" python QA.py \
    --model_name_or_path $base_model \
    --target_model $target_model \
    --segment_size $segment_size \
    --merge_size $merge_size \
    --mem_size $mem_size \
    --data_path $data_path \
    --output_dir $output_dir \
    --learning_rate $learning_rate \
    --compressor_hidden_layers $num_compressor_encoder_layers \
    --instruction_name $instruction_name \
    --random_num_documents \
    --max_num_documents $max_num_documents \
    --min_num_documents $min_num_documents \
    --num_eval_documents $max_num_documents \
    --pool_window_size $pool_window_size \
    --gold_first_for_kd \
    --random_pool_window_size \
    --remove_unused_columns False \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $batch_size \
    --max_steps $max_steps \
    --save_steps $save_steps \
    --deepspeed $deepspeed_config \
    --lr_scheduler_type "linear" \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --resume_from_checkpoint $checkpoint_path \
    --max_grad_norm 2.0 \
    --mrMR False \
    --alpha 0.001 \
    --bf16 \
    --restatement False \
    --ppl_memory False \
    --mean False \
    --mem_lora False \
    --num_mem_fusion_layers $num_mem_fusion_layers \
    --benchmark_metric $benchmark_metric \
    --post_append $is_post_append \
    --segment_size $segment_size \
    --split $is_split \
    --autoregressive $is_autoregressive \
    --is_train $is_train \
    --fine_tune $fine_tune \
    --prefix_type $prefix_type \
    --is_random $is_random \
    --report_to "none" \
    --launch_tkdr $launch_tkdr \
    --key_percentage $key_percentage \
    --merge_sizes $merge_sizes \
    --adaptive_pick $adaptive_pick \
    --tau $tau \
    --lamda_select $lamda_select \
    --lamda_merge $lamda_merge \
    --coarse_grained_on $coarse_grained_on \
    --fine_grained_on $fine_grained_on \
    --redun_coarse $redun_coarse \
    --redun_fine $redun_fine \
    --lora_the_encoder $lora_the_encoder \
    --max_doc_tokens $max_doc_tokens \
    --draw $draw \
    --enable_encore $enable_encore \
    --top_p $top_p \
    --uniform_distribution $uniform_distribution \
    --top_p_list $top_p_list \
    | tee ${output_dir}/train_restatement.log