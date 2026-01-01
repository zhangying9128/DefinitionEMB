#!/usr/bin/env bash

python fairseq_code/definitionemb_scripts/rouge_significance.py --reference-file /path/to/public_data/dataset-name/test.target --baseline-files /path/to/saved_models/baseline/experiment/test.hypo  --new-files /path/to/saved_models/new/experiment/test.hypo --metric rouge-1

