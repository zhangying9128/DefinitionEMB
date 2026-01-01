#!/usr/bin/env bash

TOTAL_NUM_UPDATES=12000    # Total number of training steps 
PEAK_LR=1e-3           # Peak learning rate, adjust as needed
MAX_TOKENS=4096        # Number of sequences per batch (batch size)
UPDATE_FREQ=4          # Increase the batch size x times

DATA_DIR=/path/to/fairseq-data-bin/dataset-name
T5_PATH=/path/to/model/t5gemma-l-ul2

SAVE_PATH=/path/to/saved_models/cnndm/
mkdir -p $SAVE_PATH

CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train $DATA_DIR \
    --seed $SEED \
    --restore-file $T5_PATH \
    --no-epoch-checkpoints \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --task summarization -s source -t target --criterion cross_entropy_datamap \
    --arch hf_t5gemma_l  \
    --max-source-positions 512 \
    --max-target-positions 512 \
    --truncate-source \
    --tie-word-embeddings \
    --used-dictionary "t5gemma" \ 
    --optimizer adafactor \
    --lr $PEAK_LR \
    --max-tokens $MAX_TOKENS \
    --save-dir $SAVE_PATH \
    --save-svd --save-svd-interval-updates 100 \
    --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --max-update $TOTAL_NUM_UPDATES --log-format simple --log-interval 200 > $SAVE_PATH/log.txt