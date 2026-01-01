#!/usr/bin/env bash

TOTAL_NUM_UPDATES=20000    # Total number of training
PEAK_LR=3e-05           # Peak learning rate, adjust as needed
MAX_TOKENS=8192        # Number of sequences per batch (batch size)
UPDATE_FREQ=8          # Increase the batch size x times

DATA_DIR=/path/to/fairseq-data-bin/dataset-name
BART_PATH=/path/to/model/bart/bart.large

SAVE_PATH=/path/to/saved_models/cnndm/

mkdir -p $SAVE_PATH

CUDA_VISIBLE_DEVICES=0 fairseq-train $DATA_DIR \
    --restore-file $BART_PATH \
    --no-epoch-checkpoints \
    --required-batch-size-multiple 1 \
    --reset-optimizer --reset-dataloader --reset-meters \ 
    --task summarization -s source -t target --criterion cross_entropy_datamap \
    --arch bart_large  \
    --max-source-positions 1024 \
    --max-target-positions 1024 \
    --truncate-source \
    --share-all-embeddings \
    --dropout 0.1 --attention-dropout 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-08 --weight-decay 0.01 --clip-norm 0.1 \
    --lr $PEAK_LR --lr-scheduler polynomial_decay --warmup-updates 500 --total-num-update $TOTAL_NUM_UPDATES \
    --max-tokens $MAX_TOKENS \
    --save-dir $SAVE_PATH \
    --save-svd --save-svd-interval-updates 100 \
    --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --max-update $TOTAL_NUM_UPDATES --log-format simple --log-interval 400 > $SAVE_PATH/log.txt

done
