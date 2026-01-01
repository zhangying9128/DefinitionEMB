#!/usr/bin/env bash
fairseq-preprocess \
    --source-lang source --target-lang target \
    --task summarization --used-dictionary bart \
    --trainpref /path/to/public_data/cnndm/train.bpe \
    --validpref /path/to/public_data/cnndm/valid.bpe \
    --testpref /path/to/public_data/cnndm/test.bpe \
    --srcdict /path/to/model/dict.txt \
    --tgtdict /path/to/model/dict.txt \
    --workers 60 \
    --destdir /path/to/fairseq-data-bin/dataset-name

