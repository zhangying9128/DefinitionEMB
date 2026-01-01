# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

import argparse
import os
import json
import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from sklearn.decomposition import TruncatedSVD, PCA
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

import math
from scipy.special import logsumexp
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
        z_c = logsumexp(z_c)
        z_c = np.exp(z_c)

        min_z = min(z_c, min_z)
        max_z = max(z_c, max_z)

    return round((min_z / max_z).item(), 4)

@torch.no_grad()
def compute_isotropy(V):
    return isotropy(V.to('cpu').detach().numpy().copy())

def evaluate_isotropy(word_embeddings, prefix=""):
    """
    Evaluate isotropy and norms for different frequency groups
    
    Returns:
        groups: list of (indices, name) tuples
        isotropy_scores: dictionary of isotropy scores
        norm_stats: dictionary of norm statistics
    """
    # Calculate isotropy for all embeddings
    all_isotropy = compute_isotropy(word_embeddings)
    print(f"{prefix} all_isotropy: {all_isotropy}")
    
    # Store isotropy scores and norm stats
    isotropy_scores = {"all": all_isotropy}
    
    groups = []
    
    # Simple frequency-based groups
    for split in ["freq", "medium", "rare"]:
        if split == "freq":
            idx = np.array(range(int(len(word_embeddings) * 0.3)))
        elif split == "medium":
            idx = np.array(range(int(len(word_embeddings) * 0.3), int(len(word_embeddings) * 0.8)))
        else:
            idx = np.array(range(int(len(word_embeddings) * 0.8), len(word_embeddings)))
        groups.append((idx, split))

        # Compute isotropy for this group
        isotropy_score = compute_isotropy(word_embeddings[idx])
        isotropy_scores[split] = isotropy_score
        
        print(f"{prefix} {split}: isotropy={isotropy_score}")

    return groups, isotropy_scores

def analyze_embeddings(word_embeddings, save_dir, embedding_type="encoder", model_name="", temperature_scaled=False):
    """
    Analyze a single embedding matrix (encoder or decoder)
    
    Args:
        word_embeddings: The embedding tensor
        save_dir: Directory to save results
        embedding_type: "encoder" or "decoder"
        model_name: Name of the model
        temperature_scaled: Whether temperature scaling was applied
    
    Returns:
        Dictionary containing all analysis results
    """
    print(f"\n{'='*50}")
    print(f"Analyzing {embedding_type} embeddings")
    print(f"{'='*50}")
    
    # Evaluate isotropy and norms
    groups, isotropy_results = evaluate_isotropy(
        word_embeddings, 
        prefix=f"[{embedding_type.upper()}]"
    )

def main(args):
    # Set random seed
    torch_fix_seed()
    
    # Create save directory
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model: {args.model_name}")
    
    if False:
        # Load huggingface model
        from transformers import AutoModel
        model = AutoModel.from_pretrained(args.model_name, torch_dtype=torch.float32, cache_dir=save_dir)
    else:
        # Load fairseq model
        from fairseq import hub_utils
        x = hub_utils.from_pretrained(
                args.save_dir,
                "model.pt",
                "/path/to/fairseq-data-bin/cnndm-t5gemma/",
            )
        model = x["models"][0].model.model

    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    
    # Get encoder embeddings
    encoder_embeddings = None
    decoder_embeddings = None
    
    # Try to get encoder embeddings
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'embed_tokens'):
        encoder_embeddings = model.encoder.embed_tokens.weight.data
        print(f"Found encoder embeddings with shape: {encoder_embeddings.size()}")
    elif hasattr(model, 'shared'):
        # Some T5 models have shared embeddings - treat as encoder
        encoder_embeddings = model.shared.weight.data
        print(f"Found shared embeddings (treating as encoder) with shape: {encoder_embeddings.size()}")
    elif hasattr(model, 'embeddings'):
        # For other transformer models
        encoder_embeddings = model.embeddings.word_embeddings.weight.data
        print(f"Found embeddings with shape: {encoder_embeddings.size()}")
    else:
        print("Warning: Cannot find encoder embeddings in the model")
    
    # Try to get decoder embeddings
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'embed_tokens'):
        decoder_embeddings = model.decoder.embed_tokens.weight.data
        print(f"Found decoder embeddings with shape: {decoder_embeddings.size()}")
        
        # Check if decoder embeddings are the same as encoder
        if encoder_embeddings is not None:
            are_same = (decoder_embeddings.data_ptr() == encoder_embeddings.data_ptr())
            if are_same:
                print("Note: Encoder and decoder share the same embedding matrix")
            else:
                print("Note: Encoder and decoder have separate embedding matrices")
    else:
        print("Warning: Cannot find decoder embeddings in the model")
    
    # Analyze encoder embeddings
    if encoder_embeddings is not None:
        analyze_embeddings(
            encoder_embeddings, 
            save_dir, 
            "encoder",
            args.model_name,
        )

    # Analyze decoder embeddings (only if they are different from encoder)
    if decoder_embeddings is not None:
        # Check if decoder embeddings are different from encoder
        if encoder_embeddings is None or (decoder_embeddings.data_ptr() != encoder_embeddings.data_ptr()):
            analyze_embeddings(
                decoder_embeddings, 
                save_dir, 
                "decoder",
                args.model_name,
            )
        else:
            print("\nSkipping decoder analysis (embeddings are shared with encoder)")



def cli_main():
    parser = argparse.ArgumentParser(description='Analyze word embeddings from HuggingFace models')
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/t5gemma-l-l-ul2",  # 或其他T5模型
        help="HuggingFace model name or path",
    )
    
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/path/to/model/t5gemma-l-ul2/",
        help="Directory to save results",
    )
    
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()