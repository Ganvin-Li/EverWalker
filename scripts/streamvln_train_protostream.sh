#!/usr/bin/env bash
# streamvln_train_protostream.sh
# ProtoStream持续学习训练脚本

BASE_VIDEO_FOLDER="task"
BASE_MODEL="model_zoo/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln/"
PROMPT_VERSION="qwen_1_5"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

USE_PROTOSTREAM=true
PROTO_DIM=512
SIMILARITY_THRESHOLD=0.8
MAX_PROTOTYPES=100
PROTO_MOMENTUM=0.95
DIVERSITY_WEIGHT=0.2
TEMPERATURE=0.5
HYPERNET_LORA_RANK=16
HYPERNET_LORA_ALPHA=32.0
HYPERNET_LORA_DROPOUT=0.1
HYPERNET_HIDDEN_DIM=256
HYPERNET_TARGET_MODULES="q_proj,v_proj"
LAMBDA_SP=1.0
LAMBDA_PP=0.3
LAMBDA_CP=0.2
DISTILL_TEMPERATURE=2.0
ENABLE_DISTILLATION=true
CONTINUAL_LEARNING=true
NUM_TASKS=20
NUM_TRAIN_EPOCHS=200
LEARNING_RATE=1e-5  
MM_VISION_TOWER_LR=2e-6
WARMUP_RATIO=0.1

export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export WANDB_MODE=offline

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-30000}

BASE_RUN_NAME="protostream_continual_${PROMPT_VERSION}_$(date +%m_%d)"
BASE_OUTPUT_DIR="checkpoints/${BASE_RUN_NAME}"

mkdir -p ${BASE_OUTPUT_DIR}

for TASK_ID in $(seq 0 $((NUM_TASKS - 1))); do
    
    SCENE_IDX=$((TASK_ID / 4))
    ENV_IDX=$((TASK_ID % 4))
    SCENE=${SCENES[$SCENE_IDX]}
    ENV=${ENVS[$ENV_IDX]}
    TASK_VIDEO_FOLDER="${BASE_VIDEO_FOLDER}/Task_$((TASK_ID + 1))"
    MASTER_PORT=$((MASTER_PORT_BASE + TASK_ID))
    TASK_OUTPUT_DIR="${BASE_OUTPUT_DIR}"
    RUN_NAME="${BASE_RUN_NAME}_task_${TASK_ID}_${SCENE}_${ENV}"
    if [ ${TASK_ID} -eq 0 ]; then
        PRETRAINED_CHECKPOINT=""
        LOAD_PROTOTYPES_FROM=""
        echo "Starting from base model"
    else
        PREV_TASK_ID=$((TASK_ID - 1))
        PREV_CHECKPOINT="${BASE_OUTPUT_DIR}/task_${PREV_TASK_ID}/task_${PREV_TASK_ID}"
        PRETRAINED_CHECKPOINT="--pretrained_checkpoint_path ${PREV_CHECKPOINT}"
        LOAD_PROTOTYPES_FROM="--load_prototypes_from ${PREV_CHECKPOINT}/prototype_lib.pt"
        echo "Loading from previous task: ${PREV_CHECKPOINT}"
    fi
    
    torchrun \
      --nnodes 1 \
      --nproc_per_node ${NUM_GPUS} \
      --rdzv_id ${RUN_NAME} \
      --rdzv_backend c10d \
      --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
      streamvln/streamvln_train.py \
        --deepspeed scripts/zero2.json \
        --model_name_or_path ${BASE_MODEL} \
        ${PRETRAINED_CHECKPOINT} \
        --use_protostream ${USE_PROTOSTREAM} \
        --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
        --proto_dim ${PROTO_DIM} \
        --similarity_threshold ${SIMILARITY_THRESHOLD} \
        --max_prototypes ${MAX_PROTOTYPES} \
        --proto_momentum ${PROTO_MOMENTUM} \
        --diversity_weight ${DIVERSITY_WEIGHT} \
        --temperature ${TEMPERATURE} \
        --hypernet_lora_rank ${HYPERNET_LORA_RANK} \
        --hypernet_lora_alpha ${HYPERNET_LORA_ALPHA} \
        --hypernet_lora_dropout ${HYPERNET_LORA_DROPOUT} \
        --hypernet_hidden_dim ${HYPERNET_HIDDEN_DIM} \
        --hypernet_target_modules ${HYPERNET_TARGET_MODULES} \
        --lambda_sp ${LAMBDA_SP} \
        --lambda_pp ${LAMBDA_PP} \
        --lambda_cp ${LAMBDA_CP} \
        --distill_temperature ${DISTILL_TEMPERATURE} \
        --enable_distillation ${ENABLE_DISTILLATION} \
        --task_sequence "${TASK_SEQUENCE}" \
        --continual_learning ${CONTINUAL_LEARNING} \
        --num_tasks ${NUM_TASKS} \
        --current_task_id ${TASK_ID} \
        --protostream_checkpoint_dir ${BASE_OUTPUT_DIR} \
        --save_prototypes true \
        ${LOAD_PROTOTYPES_FROM} \
        --log_prototype_stats true \
        --prototype_log_interval 100 \
        --version ${PROMPT_VERSION} \
        --video_folder "${TASK_VIDEO_FOLDER}" \
        --group_by_task False \
        --num_history 8 \
        --num_future_steps 4 \
        --num_frames 32 \
        --data_augmentation True \
        --vision_tower ${VISION_MODEL_VERSION} \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio anyres_max_9 \
        --image_grid_pinpoints "(1x1),...,(6x6)" \
        --bf16 True \
        --run_name ${RUN_NAME} \
        --output_dir "${BASE_OUTPUT_DIR}/task_${TASK_ID}" \
        --num_train_epochs ${NUM_TRAIN_EPOCHS} \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --evaluation_strategy "no" \
        --save_strategy "no" \
        --learning_rate ${LEARNING_RATE} \
        --mm_vision_tower_lr ${MM_VISION_TOWER_LR} \
        --weight_decay 0.01 \
        --warmup_ratio ${WARMUP_RATIO} \
        --lr_scheduler_type "cosine" \
        --logging_steps 10 \
        --tf32 True \
        --model_max_length 32768 \
        --gradient_checkpointing True \
        --dataloader_num_workers 8 \
        --lazy_preprocess True \
        --torch_compile False \
        --dataloader_drop_last True \
        --report_to wandb
done