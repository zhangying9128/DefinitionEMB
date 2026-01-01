#!/usr/bin/env bash

TASK="MRPC"
if [ $TASK = "RTE" ] ; then
    NUM_CLASSES=2
    LR=1e-5          # Peak learning rate, adjust as needed
    MAX_SENTENCES=32        # Batch size
    TOTAL_NUM_UPDATES=1020    # Total number of training steps 
    WARMUP_UPDATES=61
elif [ $TASK = "MRPC" ] ; then
    NUM_CLASSES=2
    LR=2e-5          # Peak learning rate, adjust as needed
    MAX_SENTENCES=64        # Batch size
    TOTAL_NUM_UPDATES=700    # Total number of training steps 
    WARMUP_UPDATES=42
fi
UPDATE_FREQ=1          # 
let LOGINTERVAL=${TOTAL_NUM_UPDATES}/50
LOGINTERVAL=${LOGINTERVAL%.*}


BART_PATH=/path/to/model/bart/bart.large/model.pt
DATA_DIR=/path/to/fairseq-data-bin/glue-bart/${TASK}-bin/

SAVE_PATH=/path/to/saved_models/glue/${TASK}/
mkdir -p $SAVE_PATH

WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 fairseq-train $DATA_DIR \
    --restore-file $BART_PATH \
    --batch-size $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction \
    --add-prev-output-tokens \
    --layernorm-embedding \
    --share-all-embeddings \
    --no-epoch-checkpoints \
    --no-last-checkpoints \
    --no-save-optimizer-state \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 \
    --arch bart_large  \
    --criterion sentence_prediction \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-08 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch 10 \
    --find-unused-parameters \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --save-dir $SAVE_PATH \
    --update-freq $UPDATE_FREQ \
    --log-format simple --log-interval $LOGINTERVAL > $SAVE_PATH/log.txt

