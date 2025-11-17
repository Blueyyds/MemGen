#!/bin/bash

export DEBUG_MODE=false
export LOG_PATH="./debug_log_2b.txt"
export CUDA_VISIBLE_DEVICES=0
export MAIN_PROCESS_PORT=29507
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1
export CUDA_VISIBLE_DEVICES=1

# options:
# - Qwen/Qwen2.5-1.5B-Instruct
# - HuggingFaceTB/SmolLM3-3B
REASONER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"   
WEAVER_MODEL="Qwen/Qwen2.5-1.5B-Instruct" 
TRIGGER_MODEL=null

# Dataset configs
DATASET_NAME="gsm8k"  # options: gsm8k, gpqa, kodcode, triviaqa
DATASET_MODE="grpo"    # options: sft or grpo

# MemGen configs
TRAIN_METHOD="grpo"    # options: sft or grpo

# Augmentation configs:
# - For gsm8k, gpqa, kodcode: MAX_PROMPT_AUG_NUM=1, MAX_INFERENCE_AUG_NUM=5
# - For triviaqa:             MAX_PROMPT_AUG_NUM=6, MAX_INFERENCE_AUG_NUM=0
MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=8
INFERENCE_LATENTS_LEN=8

# Trained weaver model path: 
# - Can point to a checkpoint directory (e.g. <output_dir>/checkpoint-100) to resume training with full state
# - Can point to a checkpoint file ending with .safetensors (e.g. <output_dir>/model.safetensors) to load weights and continue training
# - If set to "null", training starts from scratch.  
LOAD_WEAVER_PATH="results/weaver/gsm8k_grpo/Qwen/Qwen2.5-1.5B-Instruct/20251116-233637/weaver/checkpoint-500"
CURRENT_TIME=$(date +%Y%m%d-%H%M%S)
OUTPUT_DIR="results/weaver/${DATASET_NAME}_${TRAIN_METHOD}/${REASONER_MODEL}/${CURRENT_TIME}"

# Wandb configs
WANDB_PROJECT="memgen"  # wandb 项目名称
WANDB_RUN_NAME="H100_weaver_${DATASET_NAME}_${TRAIN_METHOD}_${REASONER_MODEL}_${CURRENT_TIME}"  # wandb 运行名称

# train
uv run -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    main.py \
    --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
    --options \
    model.reasoner_model_name ${REASONER_MODEL} \
    model.weaver.weaver_model_name ${WEAVER_MODEL} \
    model.trigger.trigger_model_name ${TRIGGER_MODEL} \
    model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
    model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
    model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
    model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
    model.load_model_path ${LOAD_WEAVER_PATH} \
    datasets.${DATASET_NAME}.mode ${DATASET_MODE} \
    run.mode train \
    run.save_dir ${OUTPUT_DIR} \
    run.train_weaver True \
    run.train_trigger False \
    run.train_weaver_method ${TRAIN_METHOD} \
    run.weaver.grpo.eval_strategy "'no'" \
    run.weaver.grpo.load_best_model_at_end False \
    run.generation.do_sample True \
    run.generation.temperature 1.0 \
    run.generation.max_response_length 512 \
    run.wandb_project ${WANDB_PROJECT} \
    run.wandb_run_name ${WANDB_RUN_NAME}




