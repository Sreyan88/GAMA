#!/bin/bash

export TRANSFORMERS_CACHE=/fs/gamma-projects/audio/gama/hf_cache/
export HF_DATASETS_CACHE=/fs/gamma-projects/audio/gama/hf_cache/

output_dir='/fs/gamma-projects/audio/gama/test_gama/stage1'
mkdir -p $output_dir
cp "$0" ${output_dir}/$(date +"%Y-%m-%d-%H-%M-%S").sh


torchrun --nproc_per_node=4 --master_port=1234 ../finetune.py \
    --base_model '/fs/nexus-projects/brain_project/acl_sk_24/GAMA/src/Llama-2-7b-chat-hf-qformer' \
    --data_path '/fs/gamma-projects/audio/audio_datasets/combine_cla_nexus_new.json' \
    --output_dir $output_dir \
    --batch_size 256 \
    --micro_batch_size 4 \
    --num_epochs 2 \
    --learning_rate 1e-3 \
    --cutoff_len 108 \
    --val_set_size 0 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules ['dummy'] \
    --train_on_inputs \
    --wandb_run_name ${output_dir} \
    --group_by_length \
    --save_steps 500 \
    --trainable_params qformer