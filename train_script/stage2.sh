#!/bin/bash

export TRANSFORMERS_CACHE=/fs/gamma-projects/audio/gama/hf_cache/
export HF_DATASETS_CACHE=/fs/gamma-projects/audio/gama/hf_cache/
output_dir='/fs/gamma-projects/audio/gama/new_data/stage_2_all_cla'
mkdir -p $output_dir
cp "$0" ${output_dir}/$(date +"%Y-%m-%d-%H-%M-%S").sh

torchrun --nproc_per_node=8 --master_port=1234 ../finetune.py \
    --base_model '/fs/gamma-projects/audio/gama/new_data/stage1_proj_cla/checkpoint-13000/pytorch_model.bin' \
    --data_path '/fs/gamma-projects/audio/audio_datasets/combine_cla_nexus_new.json' \
    --output_dir $output_dir \
    --batch_size 256 \
    --micro_batch_size 1 \
    --num_epochs 2 \
    --learning_rate 1e-4 \
    --cutoff_len 108 \
    --val_set_size 0 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length \
    --wandb_run_name ${output_dir} \
    --save_steps 2300 \
    --trainable_params qformer_all