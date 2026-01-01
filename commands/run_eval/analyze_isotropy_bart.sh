#!/usr/bin/env bash
SAVE_PATH=/path/to/model/bart/bart.large/

CUDA_VISIBLE_DEVICES=0 python fairseq_code/fairseq/examples/bart/analyze_isotropy.py $DATA_DIR \
  --task summarization -s source -t target \
  --model-dir $SAVE_PATH \
  --model-file model.pt
