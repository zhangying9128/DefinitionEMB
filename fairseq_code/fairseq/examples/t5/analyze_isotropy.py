# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from fairseq import tasks, options, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf

import argparse
from argparse import Namespace
from omegaconf import DictConfig

import os
import json
import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

def torch_fix_seed(seed=1234):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

@torch.no_grad()
def compute_isotropy(V):
    return isotropy(V.to('cpu').detach().numpy().copy())

import math
def isotropy(embeddings: np.ndarray) -> float:
    """
    Computes isotropy score.
    Defined in Section 5.1, equations (7) and (8) of the paper.

    Args:
        embeddings: word vectors of shape (n_words, n_dimensions)

    Returns:
        float: isotropy score
    """
    min_z = math.inf
    max_z = -math.inf

    eigen_values, eigen_vectors = np.linalg.eig(np.matmul(embeddings.T, embeddings))
    for i in range(eigen_vectors.shape[1]):
        z_c = np.matmul(embeddings, np.expand_dims(eigen_vectors[:, i], 1))
        # 防止exp溢出
        z_c = np.clip(z_c, -50, 50)  # exp(50)约等于10^21
        z_c = np.exp(z_c)
        z_c = np.sum(z_c)
        min_z = min(z_c, min_z)
        max_z = max(z_c, max_z)
    return round((min_z / max_z).item(), 4)


def evaluate_isotropy(word_embeddings):
    isotropy = compute_isotropy(word_embeddings)
    print("all_isotropy", isotropy)

    groups = []
    for split in ["freq", "medium", "rare"]:
        if split == "freq":
            idx = np.array(range(int(len(word_embeddings) * 0.3)))
        elif split == "medium":
            idx = np.array(range(int(len(word_embeddings) * 0.3), int(len(word_embeddings) * 0.8)))
        else:
            idx = np.array(range(int(len(word_embeddings) * 0.8), len(word_embeddings)))
        groups.append((idx, split))

        isotropy = compute_isotropy(word_embeddings[idx])
        print("{} : isotropy {}".format(split, isotropy))

    return groups

def main(cfg: DictConfig):
    model_dir, model_file, data_dir = cfg.model_dir, cfg.model_file, cfg.data
    mean_temparature = 400 # we need to set it for t5 models, because original embeddings are too large in magnitude for computing isotropy
    left_start, right_end = 3, -128
    torch_fix_seed()

    # Setup task, e.g., translation, language modeling, etc.
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)
    task = tasks.setup_task(cfg.task)

    # load model
    if model_dir == "pytorch/fairseq":
        model = torch.hub.load("pytorch/fairseq", model_file)
    else:
        from fairseq import hub_utils
        x = hub_utils.from_pretrained(
                model_dir,
                model_file,
                data_dir,
            )
        model = x["models"][0]

    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()#.half()

    word_embeddings = model.model.shared.weight.data
    _word_embeddings = word_embeddings[left_start: right_end]
    print("size of word_embeddings", _word_embeddings.size())

    if mean_temparature:
        temperature = 400
        _word_embeddings = _word_embeddings / temperature

    if torch.isnan(_word_embeddings).any():
        nan_rows = torch.where(torch.isnan(_word_embeddings).any(dim=1))[0]
        print(f"Found NaN in {len(nan_rows)} rows:")

    if torch.isinf(_word_embeddings).any():
        inf_rows = torch.where(torch.isinf(_word_embeddings).any(dim=1))[0]
        print(f"Found Inf in {len(inf_rows)} rows:")
        _word_embeddings = _word_embeddings.clamp(-1e10, 1e10)

    groups = evaluate_isotropy(_word_embeddings)


def cli_main():
    parser = options.get_generation_parser()
    parser.add_argument(
        "--model-dir",
        required=True,
        type=str,
        default="t5.large/",
        help="path containing model file and src_dict.txt",
    )
    parser.add_argument(
        "--model-file",
        type=str,
        default="checkpoint_best.pt",
        help="where in model_dir are weights saved",
    )
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
