#!/usr/bin/env bash
TOTAL_NUM_UPDATES=400000    # Total number of training steps
PEAK_LR=1e-4           # Peak learning rate, adjust as needed
MAX_TOKENS=4096        # Number of sequences per batch (batch size)  
UPDATE_FREQ=1          # Increase the batch size x times

DATA_DIR=/path/to/fairseq-data-bin/wikidictionary-en-roberta
RoBERTa_PATH=/path/to/model/roberta/roberta.base/

SAVE_PATH=/path/to/saved_dictemb_models/roberta/
mkdir -p $SAVE_PATH

CUDA_VISIBLE_DEVICES=0 fairseq-train $DATA_DIR \
    --restore-file $RoBERTa_PATH \
    --no-epoch-checkpoints \
    --no-last-checkpoints \
    --no-save-optimizer-state \
    --reset-optimizer --reset-dataloader --reset-meters \
    --task distill_definition -s source -t target --criterion mean_square_error \
    --arch roberta_emb --defn-strategy linear --mask-context --update-emb \
    --filter-ratio 0 \
    --max-source-positions 512 \
    --max-target-positions 512 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.1 \
    --lr $PEAK_LR --lr-scheduler inverse_sqrt --warmup-updates 24000 --warmup-init-lr '1e-07' \
    --max-tokens $MAX_TOKENS \
    --save-dir $SAVE_PATH \
    --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --batch-size-valid 1 \
    --max-update $TOTAL_NUM_UPDATES --log-format simple --log-interval 4000 > $SAVE_PATH/log.txt
