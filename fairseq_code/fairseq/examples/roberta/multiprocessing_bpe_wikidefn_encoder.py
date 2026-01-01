#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import sys
from collections import Counter
from multiprocessing import Pool

from fairseq.data.encoders.gpt2_bpe import get_encoder
import json

def main():
    """
    Helper script to encode raw text with the GPT-2 BPE using multiple processes.

    The encoder.json and vocab.bpe files can be obtained here:
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder-json",
        help="path to encoder.json",
    )
    parser.add_argument(
        "--vocab-bpe",
        type=str,
        help="path to vocab.bpe",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["-"],
        help="input files to filter/encode",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=["-"],
        help="path to save encoded outputs",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="keep empty lines",
    )
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    with contextlib.ExitStack() as stack:
        inputs = [
            stack.enter_context(open(input, "r", encoding="utf-8"))
            if input != "-"
            else sys.stdin
            for input in args.inputs
        ]
        outputs = [
            stack.enter_context(open(output, "w", encoding="utf-8"))
            if output != "-"
            else sys.stdout
            for output in args.outputs
        ]

        encoder = MultiprocessingEncoder(args)
        pool = Pool(args.workers, initializer=encoder.initializer)
        encoded_lines = pool.imap(encoder.encode_lines, zip(*inputs), 100)

        stats = Counter()
        count_line = 0
        for i, (filt, enc_lines) in enumerate(encoded_lines, start=1):
            if filt == "PASS":
                enc_lines = enc_lines[0]
                assert len(enc_lines) == len(outputs)
                if False: #for all mask
                    for j in range(len(outputs)):
                        print(enc_lines[j], file=outputs[j])
                else: #autoregressive mask
                    for j in range(len(outputs)):
                        for k in range(len(enc_lines[j])):
                            print(enc_lines[j][k], file=outputs[j])
                    count_line += len(enc_lines[0])

            else:
                stats["num_filtered_" + filt] += 1
            if i % 10000 == 0:
                print("processed {} lines".format(i), file=sys.stderr)

        for k, v in stats.most_common():
            print("[{}] filtered {} lines".format(k, v), file=sys.stderr)
        print(count_line, "lines")

class MultiprocessingEncoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        global bpe
        bpe = get_encoder(self.args.encoder_json, self.args.vocab_bpe)

    def encode(self, line):
        global bpe
        line = json.loads(line.strip())
        src = []
        tgt = []
        mask_positions = []
        for i, subline in enumerate(line):
            if i % 2 == 0:
                s = bpe.encode(subline)
                src += s
                tgt += [-1] * len(s)
            else:
                t = bpe.encode(subline[1])
                for j in range(len(t)):
                    mask_positions.append(len(tgt)+j)
                tgt += t
                #for using all types of mask
                src += bpe.encode(' '+subline[0]) * len(t)
                #for only true source
                #src += t

        sources = []
        targets = []
        sequence = src.copy()
        for p in mask_positions:
            target = [-1] * len(sequence)
            target[p] = tgt[p]
            sources.append(list(map(str, sequence.copy())))
            targets.append(list(map(str, target)))
            sequence[p] = target[p]

        #return list(map(str, src)), list(map(str, tgt))
        return sources, targets

    def decode(self, tokens):
        global bpe
        return bpe.decode(tokens)

    def encode_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        for line in lines:
            line = line.strip()
            if len(line) == 0 and not self.args.keep_empty:
                return ["EMPTY", None]
            src_tokens, tgt_tokens = self.encode(line)
            #enc_lines.append((" ".join(src_tokens), " ".join(tgt_tokens)))
            src_tokens = [" ".join(src_token) for src_token in src_tokens]
            tgt_tokens = [" ".join(tgt_token) for tgt_token in tgt_tokens]
            assert len(src_tokens) == len(tgt_tokens)
            enc_lines.append((src_tokens, tgt_tokens))

        return ["PASS", enc_lines]

    def decode_lines(self, lines):
        dec_lines = []
        for line in lines:
            tokens = map(int, line.strip().split())
            dec_lines.append(self.decode(tokens))
        return ["PASS", dec_lines]


if __name__ == "__main__":
    main()
