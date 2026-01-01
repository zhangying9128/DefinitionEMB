#!/usr/bin/env bash

TASK=RTE
if [ $TASK = "RTE" ] ; then
    NUM_CLASSES=2
    LR=2e-5          # Peak learning rate, adjust as needed
    MAX_SENTENCES=16        # Batch size
    TOTAL_NUM_UPDATES=2036    # Total number of training steps 
    WARMUP_UPDATES=122
elif [ $TASK = "MRPC" ] ; then
    NUM_CLASSES=2
    LR=1e-5          # Peak learning rate, adjust as needed
    MAX_SENTENCES=16        # Batch size
    TOTAL_NUM_UPDATES=2296    # Total number of training steps 
    WARMUP_UPDATES=137
fi

LOGINTERVAL=50
UPDATE_FREQ=1 

ROBERTA_PATH=/path/to/model/roberta/roberta.base/model.pt
DATA_DIR=/path/to/fairseq-data-bin/glue-roberta/${TASK}-bin/

SAVE_PATH=/path/to/saved_models/glue/${TASK}/
mkdir -p $SAVE_PATH

WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 fairseq-train $DATA_DIR \
    --restore-file $ROBERTA_PATH \
    --batch-size $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction \
    --no-epoch-checkpoints \
    --no-last-checkpoints \
    --no-save-optimizer-state \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 \
    --arch roberta  \
    --criterion sentence_prediction \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch 10 \
    --find-unused-parameters \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --save-dir $SAVE_PATH \
    --update-freq $UPDATE_FREQ \
    --log-format simple --log-interval $LOGINTERVAL > $SAVE_PATH/log.txt

