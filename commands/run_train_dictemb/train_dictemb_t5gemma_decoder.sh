#!/usr/bin/env bash

TOTAL_NUM_UPDATES=125000    # Total number of training steps 
PEAK_LR=1e-3           # Peak learning rate, adjust as needed
MAX_TOKENS=8192        # Number of sequences per batch (batch size)
UPDATE_FREQ=1          # Increase the batch size x times

DATA_DIR=/path/to/fairseq-data-bin/wikidictionary-en-t5gemma/
T5_PATH=/path/to/model/t5gemma-l-ul2

SAVE_PATH=/path/to/saved_dictemb_models/t5gemma-decoder/
mkdir -p $SAVE_PATH

CUDA_VISIBLE_DEVICES=0 fairseq-train $DATA_DIR \
    --restore-file $T5_PATH \
    --no-epoch-checkpoints \
    --no-last-checkpoints \
    --reset-optimizer --reset-dataloader --reset-meters \
    --task distill_definition -s source -t target --criterion mean_square_error \
    --arch t5gemma_emb_l --defn-strategy linear \
    --filter-ratio 0 \
    --max-source-positions 512 \
    --max-target-positions 512 \
    --optimizer adafactor \
    --used-dictionary "t5gemma" \ 
    --lr $PEAK_LR \
    --max-tokens $MAX_TOKENS \
    --save-dir $SAVE_PATH \
    --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --save-interval-updates 2500 \
    --keep-interval-updates 1 \
    --save-interval 1000000 \
    --validate-interval-updates 2500 \
    --validate-interval 1000000 \
    --max-update $TOTAL_NUM_UPDATES --log-format simple --log-interval 250 --target-encoder-embeddings > $SAVE_PATH/log.txt

