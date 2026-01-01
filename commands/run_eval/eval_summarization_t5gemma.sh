#!/usr/bin/env bash

SAVE_PATH=/path/to/saved_models/t5gemma/
DATA_DIR=/path/to/fairseq-data-bin/dataset-name

CUDA_VISIBLE_DEVICES=0 python fairseq_code/fairseq/examples/t5gemma/summarize.py \
  --model-dir $SAVE_PATH \
  --data-dir $DATA_DIR \
  --model-file checkpoint_best.pt \
  --bsz 4 \
  --src /path/to/public_data/dataset-name/test.source \
  --out $SAVE_PATH/test.hypo
  
files2rouge /path/to/public_data/dataset-name/test.target $SAVE_PATH/test.hypo

