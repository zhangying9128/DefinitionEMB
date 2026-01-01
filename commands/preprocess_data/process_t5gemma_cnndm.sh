#!/usr/bin/env bash
# summarization
fairseq-preprocess \
    --source-lang source --target-lang target \
    --task summarization --used-dictionary t5gemma \
    --srcdict /path/to/t5gemma-xl-ul2/dict.txt \
    --tgtdict /path/to/t5gemma-xl-ul2/dict.txt \
    --destdir /path/to/fairseq-data-bin/cnndm-t5gemma/ \
    --trainpref /path/to/public_data/cnn_dm/tokenized_t5gemma_unicode/train.tok \
    --validpref /path/to/public_data/cnn_dm/tokenized_t5gemma_unicode/valid.tok \
    --testpref /path/to/public_data/cnn_dm/tokenized_t5gemma_unicode/test.tok \
    --workers 60
