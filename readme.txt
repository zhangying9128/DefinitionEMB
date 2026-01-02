1. construct data-bin for fairseq

commands/preprocess_data/process_seq2seq_data.sh

2. finetune PLM on dictionary data to build dictemb

commands/run_train_dictemb/train_dictemb_bart.sh

3. finetune dictemb on downstream dataset, e.g., cnndm

commands/run_train_summarization/train_BART_cnndm.sh
commands/run_train_glue/train_Bart_glue.sh

4. eval fine-tuned model on downstream dataset
commands/run_eval/eval_summarization_bart.sh