#!/usr/bin/env bash
SAVE_PATH=/path/to/model/t5-large
DATA_DIR=/path/to/fairseq-data-bin/wikidictionary-en-t5

CUDA_VISIBLE_DEVICES=0 python fairseq_code/fairseq/examples/t5/analyze_isotropy.py $DATA_DIR \
  --mean-temparature \
  --task summarization -s source -t target \
  --model-dir $SAVE_PATH \
  --model-file model.pt

