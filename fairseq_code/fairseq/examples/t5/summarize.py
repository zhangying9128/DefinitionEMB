# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import argparse
from tqdm import tqdm

XSUM_KWARGS = dict(num_beams=6, max_length=60, min_length=10, no_repeat_ngram_size=3)
CNN_KWARGS = dict(num_beams=4, max_length=140, min_length=55, no_repeat_ngram_size=3, length_penalty=2.0)
BillSum_KWARGS = dict(num_beams=4, max_length=200, min_length=10, no_repeat_ngram_size=3)

from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-large', cache_dir='/path/to/model/t5-large')

def sample(slines, model, eval_kwargs):
    # convert slines to fairseq batch
    # src_tokens = tokens + eos + pad
    tokenized_lines = []
    max_len = 0
    for line in slines:
        line = tokenizer.tokenize(line)
        line = line + ['</s>']
        max_len = max(max_len, len(line))
        if len(line) > 512:
            line = line[:512]
        tokenized_lines.append(line)
    
    for line in tokenized_lines:
        line += ['<pad>'] * (max_len - len(line))
    
    src_tokens = [tokenizer.convert_tokens_to_ids(line) for line in tokenized_lines]
    src_tokens = torch.tensor(src_tokens).long()
    if torch.cuda.is_available():
        src_tokens = src_tokens.cuda()

    # generate
    with torch.no_grad():
        hypo_tokens = model.model.generate(src_tokens, output_hidden_states=False,**eval_kwargs)

    sentences = [tokenizer.decode(hypo, skip_special_tokens=True) for hypo in hypo_tokens]
    return sentences

@torch.no_grad()
def generate(model, infile, eval_kwargs, outfile="bart_hypo.txt", bsz=32, n_obs=None):
    count = 1

    # if n_obs is not None: bsz = min(bsz, n_obs)

    with open(infile, encoding="utf-8") as f:
        source = f.readlines()

    with open(outfile, "w", encoding="utf-8") as fout:
        sline = source[0].strip()
        slines = [sline]
        for i in tqdm(range(1, len(source))):
            sline = source[i].strip()
            if n_obs is not None and count > n_obs:
                break
            if count % bsz == 0:
                hypotheses_batch = sample(slines, model, eval_kwargs)
                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis + "\n")
                    fout.flush()
                slines = []

            slines.append(sline.strip())
            count += 1

        if slines != []:
            hypotheses_batch = sample(slines, model, eval_kwargs)
            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + "\n")
                fout.flush()


def main():
    """
    Usage::

         python examples/bart/summarize.py \
            --model-dir $HOME/bart.large.cnn \
            --model-file model.pt \
            --src $HOME/data-bin/cnn_dm/test.source
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        required=True,
        type=str,
        default="bart.large.cnn/",
        help="path containing model file and src_dict.txt",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=str,
        default="bart.large.cnn/",
        help="path containing data file and src_dict.txt",
    )
    parser.add_argument(
        "--model-file",
        default="checkpoint_best.pt",
        help="where in model_dir are weights saved",
    )
    parser.add_argument(
        "--src", default="test.source", help="text to summarize", type=str
    )
    parser.add_argument(
        "--out", default="test.hypo", help="where to save summaries", type=str
    )
    parser.add_argument("--bsz", default=32, help="where to save summaries", type=int)
    parser.add_argument(
        "--n", default=None, help="how many examples to summarize", type=int
    )
    parser.add_argument(
        "--xsum-kwargs",
        action="store_true",
        default=False,
        help="if true use XSUM_KWARGS else CNN_KWARGS",
    )
    parser.add_argument(
        "--billsum-kwargs",
        action="store_true",
        default=False,
        help="if true use BillSum_KWARGS else CNN_KWARGS",
    )
    args = parser.parse_args()
    if args.xsum_kwargs:
        eval_kwargs = XSUM_KWARGS
    elif args.billsum_kwargs:
        eval_kwargs = BillSum_KWARGS
    else:
        eval_kwargs = CNN_KWARGS
    if args.model_dir == "pytorch/fairseq":
        model = torch.hub.load("pytorch/fairseq", args.model_file)
    else:
        from fairseq import hub_utils
        x = hub_utils.from_pretrained(
                args.model_dir,
                args.model_file,
                args.data_dir,
            )
        model = x["models"][0]
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()#.half()

    generate(
        model, args.src, bsz=args.bsz, n_obs=args.n, outfile=args.out, eval_kwargs=eval_kwargs
    )


if __name__ == "__main__":
    main()
