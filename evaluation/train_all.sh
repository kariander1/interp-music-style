#!/bin/bash

SOURCE_DIR="data/musicTI_dataset/images/timbre"
MODEL_CKPT="models/ldm/sd/model.ckpt"
CONFIG_PATH="configs/stable-diffusion/v1-finetune.yaml"
LOG_DIR="logs"

mkdir -p "$LOG_DIR"

for DIR in "$SOURCE_DIR"/*/; do
    DIR_NAME=$(basename "$DIR")
    JOB_NAME="train_${DIR_NAME}"

    OUT_LOG="${LOG_DIR}/${DIR_NAME}.out"
    ERR_LOG="${LOG_DIR}/${DIR_NAME}.err"

    echo "Running training for: $DIR_NAME"
    echo "Logs: stdout -> $OUT_LOG, stderr -> $ERR_LOG"

    echo "Running on node: $(hostname)" | tee -a "$OUT_LOG"
    echo "Current Python path: $(which python)" | tee -a "$OUT_LOG"
    python --version | tee -a "$OUT_LOG"

    python main.py \
        --base "${CONFIG_PATH}" -t \
        --actual_resume "${MODEL_CKPT}" \
        -n "${DIR_NAME}" \
        --gpus 0, \
        --data_root "${DIR}" \
        >> "$OUT_LOG" 2>> "$ERR_LOG"

    echo "Completed training for $DIR_NAME"
done
