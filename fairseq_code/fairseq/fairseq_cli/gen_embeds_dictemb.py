# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pickle
import os
import torch
from fairseq import utils 
from fairseq.models.bart import BARTModel
from transformers import BartTokenizer

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

@torch.no_grad()
def generate(bart, infile, outfile, component, layer, save_path, bsz=32,):
    count = 0

    token_dict = {}
    slines, olines = [], []
    with open(infile, encoding="utf-8") as source, open(outfile, encoding="utf-8") as output:
        source_data = source.readlines()
        output_data = output.readlines()
    
    assert len(source_data) == len(output_data)

    if layer == 0:
        if component == 2:
            embed_tokens = bart.model.decoder.embed_tokens
            dictionary = bart.task.target_dictionary
        elif component == 1:
            embed_tokens = bart.model.encoder.embed_tokens
            dictionary = bart.task.source_dictionary

        for key in range(len(dictionary)):
            batch_idx = x_id = key
            token = tokenizer.convert_ids_to_tokens(key)

            if token != "<pad>":
                token_dict[token] = token_dict.get(token, []) + [(embed_tokens.weight[key].detach().cpu().numpy(), batch_idx, x_id, key)] # d[token] = [(embedding, batch_idx, x_id, key)]


    else:
        batch_idx = 0
        for i in range(len(source_data)):
            slines.append(source_data[i].strip())
            olines.append(output_data[i].strip())

            if (len(slines) == bsz) or i == len(source_data) - 1:
                tokenized_src_sentences = [bart.encode(sentence) for sentence in slines]
                tokenized_out_sentences = [bart.encode(sentence)[1:] for sentence in olines]

                batch_iterator = bart.task.get_batch_iterator(
                        dataset=bart.task.build_dataset_fromsrctgt_for_inference(tokenized_src_sentences, tokenized_out_sentences),
                        max_tokens=bart.cfg.dataset.max_tokens,
                        max_sentences=bart.cfg.dataset.batch_size,
                        max_positions=bart.max_positions,
                        ignore_invalid_inputs=False,
                        disable_iterator_cache=True,
                    ).next_epoch_itr(shuffle=False)


                for batch in batch_iterator:
                    batch = utils.move_to_cuda(batch)

                    # forward computing
                    with torch.no_grad():
                        net_output = bart.model(**batch["net_input"])
                    #bart decoder states


                    if component == 2:
                        hidden_states = net_output[1]["inner_states"]
                        tokens = batch["target"]
                    elif component == 1:
                        hidden_states = net_output[1]["encoder_states"]
                        tokens = batch["net_input"]["src_tokens"]

                    # obtain layer's hidden states:
                    hidden_y = hidden_states[layer].transpose(0, 1) # batch x len x dim
                    bsz = tokens.size()

                    for x_id in range(bsz[1]):
                        y = hidden_y[:, x_id]
                        cnt = 0
                        for b in range(bsz[0]):
                            key = tokens[b, x_id].detach().cpu().item()
                            token = tokenizer.convert_ids_to_tokens(key)
                            if token != "<pad>":
                                token_dict[token] = token_dict.get(token, []) + [(y[cnt].detach().cpu().numpy(), batch_idx, x_id, key)] # d[token] = [(embedding, batch_idx, x_id, key)]
                            cnt += 1

                    batch_idx += 1


                slines = []
                olines = []

    component = "encoder" if component==1 else "decoder"
    print("start to save hidden_states")
    pickle.dump((batch_idx, token_dict), open(os.path.join(save_path, "{}.layer.{}.dict".format(component, layer)), 'wb'))




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
        "--out", default="test.output", help="output summarization", type=str
    )
    parser.add_argument(
        "--save-path", default="test.output", help="save hidden_states in layer", type=str
    )
    parser.add_argument("--bsz", default=32, help="where to save summaries", type=int)
    parser.add_argument("--component", default=1, help="1:encoder, 2:decoder", type=int)
    parser.add_argument("--layer", default=0, help="the layer to analyze", type=int)
    args = parser.parse_args()
    if args.model_dir == "pytorch/fairseq":
        bart = torch.hub.load("pytorch/fairseq", args.model_file)
    else:
        bart = BARTModel.from_pretrained(
            args.model_dir,
            checkpoint_file=args.model_file,
            data_name_or_path=args.data_dir,
        )

    bart = bart.eval()
    if torch.cuda.is_available():
        bart = bart.cuda()#.half()

    generate(
        bart, args.src, args.out, args.component, args.layer, args.save_path, bsz=args.bsz,
    )


if __name__ == "__main__":
    main()
