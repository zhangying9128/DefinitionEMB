#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, os, torch, random
import torch.nn.functional as F
from tqdm import tqdm
from omegaconf import OmegaConf
from collections import defaultdict
from fairseq import utils, tasks
from fairseq.models.bart import BARTModel
from fairseq.models.roberta import RobertaModel
from sklearn.decomposition import PCA
import argparse

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='Remove PCA components and optionally apply z-score normalization')
parser.add_argument('--model', type=str, default='bart', choices=['bart', 'roberta', 't5', 't5gemma'],
                    help='Model type to process')
parser.add_argument('--remove-pca', action='store_true', default=True,
                    help='Whether to remove PCA components (default: True)')
parser.add_argument('--n-components', type=int, default=10,
                    help='Number of PCA components to remove (default: 10)')
parser.add_argument('--apply-zscore', action='store_true', default=False,
                    help='Whether to apply z-score normalization after PCA removal')
parser.add_argument('--zscore-dim', type=int, default=0,
                    help='Dimension along which to compute z-score (0: across vocab, 1: across dimensions)')
parser.add_argument('--eps', type=float, default=1e-6,
                    help='Small epsilon value to prevent division by zero in z-score')

args = parser.parse_args()

# 根据参数设置模型和路径
baseline_model = args.model

if baseline_model == 't5gemma':
    pretrained_dir = "/path/to/model/t5gemma-l-ul2"
    base_save_name = "T5Gemma-large"
elif baseline_model == "bart":
    pretrained_dir = "/path/to/model/bart/bart.large"
    base_save_name = "Bart-large"
elif baseline_model == "roberta":
    pretrained_dir = "/path/to/model/roberta/roberta.base"
    base_save_name = "RoBERTa-base"
elif baseline_model == "t5":
    pretrained_dir = "/path/to/model/t5-large"
    base_save_name = "T5-large"

# 根据操作构建保存目录名
save_dir_parts = []
if args.remove_pca:
    save_dir_parts.append(f"RemovePCA{args.n_components}")
if args.apply_zscore:
    save_dir_parts.append(f"ZScoreNorm_dim{args.zscore_dim}")
    
if save_dir_parts:
    save_dir = "_".join(save_dir_parts) + f"_Pretrained{base_save_name}"
else:
    save_dir = f"Pretrained{base_save_name}_NoProcessing"

print(f"Save directory: {save_dir}")
print(f"Operations to apply: Remove PCA={args.remove_pca}, Z-Score={args.apply_zscore}")


def remove_pca_components(weight, n_components=10):
    """
    移除权重矩阵的均值和主成分
    
    Args:
        weight: 输入的权重张量
        n_components: PCA组件数量
    
    Returns:
        处理后的权重张量
    """
    # 计算均值并中心化
    mu = torch.mean(weight, dim=0).unsqueeze(0)
    temp = weight - mu
    
    # 转换为numpy进行PCA
    word_emb_matrix = temp.to('cpu').detach().numpy().copy()
    
    # PCA拟合和变换
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(word_emb_matrix)
    
    # 获取主成分轴
    principalAxes = pca.components_
    principalAxes = weight.new_tensor(principalAxes)
    
    # 从中心化的权重中减去主成分投影
    toSubstract = torch.mm(torch.mm(weight, principalAxes.t()), principalAxes)
    weight_processed = temp - toSubstract
    
    return weight_processed, pca

def apply_zscore_normalization(weight, dim=0, eps=1e-6):
    """
    对权重矩阵应用z-score normalization
    
    Args:
        weight: 输入的权重张量 [vocab_size, embed_dim]
        dim: 沿着哪个维度进行标准化
             0: 对每个词汇进行标准化（跨embedding维度）
             1: 对每个维度进行标准化（跨词汇）
        eps: 防止除零的小值
    
    Returns:
        标准化后的权重张量
    """
    print(f"Applying z-score normalization along dimension {dim}")
    
    # 计算均值和标准差
    mean = torch.mean(weight, dim=dim, keepdim=True)
    std = torch.std(weight, dim=dim, keepdim=True)
    
    # 防止除零
    std = torch.where(std < eps, torch.ones_like(std) * eps, std)
    
    # 应用z-score标准化
    weight_normalized = (weight - mean) / std
    
    # 打印统计信息
    print(f"  Before z-score - Mean: {weight.mean().item():.6f}, Std: {weight.std().item():.6f}")
    print(f"  After z-score - Mean: {weight_normalized.mean().item():.6f}, Std: {weight_normalized.std().item():.6f}")
    
    return weight_normalized

def process_embeddings(weight, remove_pca=True, n_components=10, apply_zscore=False, zscore_dim=1, eps=1e-6):
    """
    处理embedding权重的统一函数
    
    Args:
        weight: 输入权重
        remove_pca: 是否移除PCA组件
        n_components: PCA组件数量
        apply_zscore: 是否应用z-score标准化
        zscore_dim: z-score标准化的维度
        eps: z-score的epsilon值
    
    Returns:
        处理后的权重和PCA对象（如果应用了PCA）
    """
    pca = None
    weight_processed = weight.clone()
    
    # Step 1: Remove PCA components if requested
    if remove_pca:
        print(f"Removing {n_components} PCA components...")
        weight_processed, pca = remove_pca_components(weight_processed, n_components)
        print(f"PCA removal completed. Explained variance ratio: {pca.explained_variance_ratio_[:5]}")
    
    # Step 2: Apply z-score normalization if requested
    if apply_zscore:
        weight_processed = apply_zscore_normalization(weight_processed, dim=zscore_dim, eps=eps)
    
    return weight_processed, pca

# current time
import datetime
now = datetime.datetime.now()
if baseline_model == 'bart':
    BART = BARTModel.from_pretrained(pretrained_dir, checkpoint_file="model.pt")
    
    model = BART.model

elif baseline_model == 'roberta':
    RoBERTa = RobertaModel.from_pretrained(pretrained_dir, checkpoint_file="model.pt")
    model = RoBERTa.model

elif baseline_model == 't5':
    from fairseq import hub_utils
    x = hub_utils.from_pretrained(
            pretrained_dir,
            "model.pt",
            "/path/to/fairseq-data-bin/dataset-name",
        )
    model = x["models"][0]

elif baseline_model == 't5gemma':
    from fairseq import hub_utils
    x = hub_utils.from_pretrained(
            pretrained_dir,
            "model.pt",
            "/path/to/fairseq-data-bin/dataset-name",
        )
    model = x["models"][0]

model.eval()
with torch.no_grad():
    if baseline_model == 'bart':
        weight = model.encoder.embed_tokens.weight.data.clone().detach()
        weight_processed, pca = process_embeddings(
            weight, 
            remove_pca=args.remove_pca,
            n_components=args.n_components,
            apply_zscore=args.apply_zscore,
            zscore_dim=args.zscore_dim,
            eps=args.eps
        )
    elif baseline_model == 'roberta':
        weight = model.encoder.sentence_encoder.embed_tokens.weight.data.clone().detach()
        weight_processed, pca = process_embeddings(
            weight, 
            remove_pca=args.remove_pca,
            n_components=args.n_components,
            apply_zscore=args.apply_zscore,
            zscore_dim=args.zscore_dim,
            eps=args.eps
        )

    elif baseline_model == 't5':
        weight = model.model.shared.weight.data.clone().detach()
        weight_processed, pca = process_embeddings(
            weight, 
            remove_pca=args.remove_pca,
            n_components=args.n_components,
            apply_zscore=args.apply_zscore,
            zscore_dim=args.zscore_dim,
            eps=args.eps
        )

    elif baseline_model == 't5gemma':
        # T5Gemma需要分别处理encoder和decoder
        print("Processing encoder embeddings...")
        encoder_weight = model.model.model.encoder.embed_tokens.weight.data.clone().detach()
        encoder_weight_new, pca_encoder = process_embeddings(
            encoder_weight,
            remove_pca=args.remove_pca,
            n_components=args.n_components,
            apply_zscore=args.apply_zscore,
            zscore_dim=args.zscore_dim,
            eps=args.eps
        )

        print("Processing decoder embeddings...")
        decoder_weight = model.model.model.decoder.embed_tokens.weight.data.clone().detach()
        decoder_weight_new, pca_decoder = process_embeddings(
            decoder_weight,
            remove_pca=args.remove_pca,
            n_components=args.n_components,
            apply_zscore=args.apply_zscore,
            zscore_dim=args.zscore_dim,
            eps=args.eps
        )


    state_dict = model.state_dict()
    if baseline_model == 'bart':
        state_dict['encoder.embed_tokens.weight'] = weight_processed
    elif baseline_model == 'roberta':
        state_dict['encoder.sentence_encoder.embed_tokens.weight'] = weight_processed
    elif baseline_model == 't5':
        state_dict['model.shared.weight'] = weight_processed
    elif baseline_model == 't5gemma':
        state_dict['model.model.encoder.embed_tokens.weight'] = encoder_weight_new
        state_dict['model.model.decoder.embed_tokens.weight'] = decoder_weight_new

    model.load_state_dict(state_dict)
    print("remove same common vector and dominating directions")

checkpoint = torch.load(os.path.join(pretrained_dir, "model.pt"), weights_only=False)
checkpoint['model'] = model.state_dict()

# print used time
used_time = datetime.datetime.now() - now
print("used time: ", used_time)
# 保存模型
if True:
    save_path = os.path.join(pretrained_dir, save_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(checkpoint, os.path.join(save_path, "model.pt"))
    print(f"Model saved to: {save_path}")
    
    # 保存处理参数信息
    params_info = {
        'model': baseline_model,
        'remove_pca': args.remove_pca,
        'n_components': args.n_components if args.remove_pca else None,
        'apply_zscore': args.apply_zscore,
        'zscore_dim': args.zscore_dim if args.apply_zscore else None,
        'eps': args.eps if args.apply_zscore else None,
        'processing_time': str(used_time)
    }
    
    with open(os.path.join(save_path, "processing_params.json"), 'w') as f:
        json.dump(params_info, f, indent=2)
    print(f"Processing parameters saved to: {os.path.join(save_path, 'processing_params.json')}")