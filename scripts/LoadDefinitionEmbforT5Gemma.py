#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, os, torch, random
import torch.nn.functional as F
from tqdm import tqdm
from omegaconf import OmegaConf
from collections import defaultdict
from fairseq import utils, tasks

def load_and_replace_embeddings(
    model,
    pretrained_model, 
    wikien_dataset,
    task,
    nodefn,
    skip,
    visible_idx,
    embed_type="encoder"  # "encoder" or "decoder"
):
    """
    通用函数：加载并替换encoder或decoder的embeddings
    
    Args:
        base_model: 基础模型
        distilling_model: 蒸馏模型
        wikien_dataset: wiki数据集
        task: 任务
        X: 要替换的vocab索引集合
        nodefn: 无定义的token集合
        skip: 跳过的token集合
        visible_idx: 可见的数据集索引
        filter_strategy: 过滤策略
        max_idx: 最大索引
        embed_type: "encoder" 或 "decoder"
    
    Returns:
        weight: 更新后的权重
    """
    
    # 根据embed_type获取对应的embedding层
    if embed_type == "encoder":
        base_embed = model.model.model.encoder.embed_tokens
    elif embed_type == "decoder":
        base_embed = model.model.model.decoder.embed_tokens
    else:
        raise ValueError(f"embed_type must be 'encoder' or 'decoder', got {embed_type}")
    
    print(f"\nProcessing {embed_type} embeddings...")
        
    # 初始化控制变量
    # wait: 标记哪些token还需要被替换（1表示需要，0表示不需要）
    wait = torch.ones(len(task.target_dictionary))
    # wait_loss: 记录每个token的重构损失
    wait_loss = torch.zeros(len(task.target_dictionary))
    # freq: 记录每个token出现的频率
    freq = torch.zeros(len(task.target_dictionary))

    # 将无定义和跳过的token标记为不需要替换
    for idx in nodefn | skip:
        wait[idx] = 0
        freq[idx] = 1 # 设置为1避免除零错误

    # 如果有GPU可用，将模型移到GPU
    if torch.cuda.is_available(): 
        pretrained_model = pretrained_model.to(device="cuda")
    
    with torch.no_grad():
        # 克隆基础模型的embedding权重
        weight = base_embed.weight.clone().detach()
        visible_idx = list(visible_idx)
        bsz = 5 # 批处理大小
        samples = []

        # 遍历所有可见的数据集索引
        for idx in tqdm(visible_idx):
            sample = wikien_dataset[idx]
            samples.append(sample)

            # 当收集够一个批次或到达最后一个样本时，进行处理
            if (len(samples) == bsz) or (idx == visible_idx[-1]):
                # 整理批次数据
                samples = wikien_dataset.collater(samples)

                # 获取原始embeddings
                old_emb = base_embed(samples['target'])

                # 将数据移到GPU（如果可用）
                if torch.cuda.is_available(): 
                    samples = utils.move_to_cuda(samples)

                # 使用训练模型获取新的embeddings
                new_emb, _ = pretrained_model(**samples["net_input"], features_only=True)
                new_emb = new_emb.detach().cpu()

                # 计算重构损失（MSE）
                loss = F.mse_loss(
                        new_emb,
                        old_emb,
                        reduction="none",
                    ).sum(-1)

                # 按ID排序以保持一致性
                _, sort_order = samples["id"].sort(descending=False)

                # 处理批次中的每个样本
                for j in sort_order:
                    _new_emb, _loss, _target = new_emb[j], loss[j], samples["target"][j].detach().cpu()

                    # 使用target mask选择有效的embeddings
                    _target_mask = samples["target_mask"][j].detach().cpu()
                    _new_emb = _new_emb[_target_mask]
                    _target = _target[_target_mask]
                    _loss = _loss[_target_mask]

                    # 根据加载策略更新权重
                    # 只替换还未处理的token
                    tgt_mask = wait[_target].bool()
                    weight[_target[tgt_mask]] = _new_emb[tgt_mask]
                    wait[_target[tgt_mask]] = 0
                    wait_loss[_target[tgt_mask]] = _loss[tgt_mask]
                    freq[_target[tgt_mask]] += 1

                # 清空批次缓存
                samples = []

        # 计算平均损失和权重
        wait_loss = wait_loss / freq
        freq = freq.unsqueeze(1)
        weight = weight / freq

        # 可选：保存重构损失到文件
        if False:
            _, sort_order = wait_loss.sort(descending=True)

            with open(os.path.join(pretrained_dir, save_dir, "reconstruction_loss.txt"), "w") as f:
                f.write("idx token Descending_loss" + "\n")
                for idx in sort_order:
                    line = [str(idx.item()), task.target_dictionary[idx], str(wait_loss[idx].item())]
                    f.write(' '.join(line) + "\n")
        return weight
    

# ======================== 主程序开始 ========================
# 配置参数
baseline_model = 't5gemma' # roberta bart
task = "cnndm"
dict_data = "cnn_dm"
pretrained_dir = "/path/to/model/t5gemma-l-ul2"
FormDictEMBtoT5Gemma = True # if true, we first load full DefinitionEMB, else, we load specific vocabs from full DefinitionEMB

# 根据条件选择不同的配置
if FormDictEMBtoT5Gemma:
    filter_strategy = "maxidx" # 
    reconstruct_ratio = 1
    distilling_model_dir = "/path/to/saved_dictemb_models/t5gemma-encoder/"
    distilling_decoder_model_dir = "/path/to/saved_dictemb_models/t5gemma-decoder/"
    save_dir = pretrained_dir +"/LoadedFullDefinitionEMB/"
else:
    filter_strategy = "task_specific" # 
    reconstruct_ratio = 1 # max as 1
    distilling_model_dir = f"{pretrained_dir}/LoadedDefinitionEMB/"
    save_dir = pretrained_dir +"/{}_Specific_Loaded{}_DefinitionEMB".format(task, reconstruct_ratio*100, task.upper())

os.makedirs(os.path.join(save_dir), exist_ok=True)

# current time
import datetime
now = datetime.datetime.now()

# ======================== 模型加载部分 ========================
if baseline_model == 't5gemma':
    from fairseq import hub_utils
    # 加载基础模型
    x = hub_utils.from_pretrained(
            pretrained_dir,
            "model.pt",
            "/path/to/fairseq-data-bin/wikidictionary-en-t5gemma",
        )
    model = x["models"][0]

    # 冻结基础模型参数
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
    print("Load Baseline Model from {}".format(os.path.join(pretrained_dir, "model.pt")))

    # 加载预训练的encoder模型
    checkpoint_file = "model.pt" if filter_strategy=="task_specific" else "checkpoint_best.pt"
    x = hub_utils.from_pretrained(
            distilling_model_dir,
            checkpoint_file,
            "/path/to/fairseq-data-bin/wikidictionary-en-t5gemma",
        )
    pretrained_model, pretrained_args = x["models"][0], x["args"]

    for params in pretrained_model.parameters():
        params.requires_grad = False
    pretrained_model.eval()  # disable dropout (or leave in train mode to finetune)
    print("Load dictEMB Model from {}".format(os.path.join(distilling_model_dir, checkpoint_file)))
 
    if FormDictEMBtoT5Gemma:
        # 加载预训练的decoder模型
        x = hub_utils.from_pretrained(
                distilling_decoder_model_dir,
                "checkpoint_best.pt",
                "/path/to/fairseq-data-bin/wikidictionary-en-t5gemma",
            )
        pretrained_decoder_model = x["models"][0]

        for params in pretrained_decoder_model.parameters():
            params.requires_grad = False
        pretrained_decoder_model.eval()  # disable dropout (or leave in train mode to finetune)
        print("Load dictEMB Decoder Model from {}".format(os.path.join(distilling_decoder_model_dir, "checkpoint_best.pt")))

    # ======================== 数据准备部分 ========================
    visible_idx = set()

    # 配置任务
    conf = OmegaConf.create({'_name': 'distill_definition', 'continue_autoregremask': False, 'filter_ratio': 0.0, 'filter_ratio_inference': 0.0, 'All_to_P': False, 'mask_context': False, 'filter_strategy': 'maxidx', 'pegasus_dictionary': False, 'data': '/path/to/fairseq-data-bin/wikidictionary-en-t5gemma', 'source_lang': 'source', 'target_lang': 'target', 'left_pad_source': True, 'left_pad_target': False, 'max_source_positions': 512, 'max_target_positions': 512, 'truncate_source': False, 'num_batch_buckets': 0, 'train_subset': 'train', 'dataset_impl': None, 'required_seq_len_multiple': 1})

    # 设置任务和加载数据集
    task = tasks.setup_task(conf)
    wikien_dataset = task.load_dataset_from_path("train", "/path/to/fairseq-data-bin/wikidictionary-en-t5gemma", epoch=1, filter_strategy="maxidx")

    # 计算要替换的token数量阈值
    max_idx = int((1 - reconstruct_ratio) * len(task.target_dictionary))

    # ======================== 词汇表处理部分 ========================
    # 初始化集合
    X = set() # 要替换的token集合
    nodefn = set() # 无定义的token集合
    skip = set() # 要跳过的token集合
    without_train = set() # 无训练数据的token集合

    if filter_strategy in ["task_specific"]:
        # load task_specific training data vocab
        with open("/path/to/public_data/{}/tokenized_t5gemma_unicode/train_word_freq.txt".format(dict_data)) as f:
            freq_data = json.loads(f.readlines()[0])

        # 建立索引到顺序的映射
        idxs2order = {}  # 较低的order应该被跳过
        order2idxs = {}
        for key in range(len(task.source_dictionary)):
            if str(key) not in freq_data:
                idxs2order[key] = len(without_train)
                order2idxs[len(without_train)] = key
                without_train.add(key)
    else:
        idxs2order = defaultdict(int)
        for i in range(len(task.target_dictionary)):
            idxs2order[i] = i

    # 读取wikidictionary token频率文件
    #tokens in frequency_fairseqidx.txt corredsponds to PLM dict.txt and excludes <pad> </s> <unk>
    with open("/path/to/public_data/wikidictionary-en/t5gemma/frequency_fairseqidx.txt") as f:
        data = json.loads(f.readlines()[0].strip())

    # 第一遍处理：识别无定义的token
    for item in data:
        fairseqidx, dataset_idxes = int(item[0]), item[1]
        if len(dataset_idxes) == 0:
            nodefn.add(int(fairseqidx))
            if filter_strategy not in ["task_specific"]:
                if idxs2order[int(fairseqidx)] >= max_idx:
                    max_idx -= 1

            else:
                if int(fairseqidx) not in idxs2order:
                    order = len(idxs2order)
                    idxs2order[int(fairseqidx)] = order
                    assert order not in order2idxs
                    order2idxs[order] = int(fairseqidx)

    if filter_strategy in ["task_specific"]:
        max_idx = max(max_idx, len(idxs2order))
        for i, (k,v) in enumerate(freq_data.items()):
            if int(k) not in idxs2order:
                order = len(idxs2order)
                idxs2order[int(k)] = order
                order2idxs[order] = int(k)

    # 第二遍处理：分类token
    for item in data:
        fairseqidx, dataset_idxes = int(item[0]), item[1]
        order = idxs2order[int(fairseqidx)]
        if int(fairseqidx) not in nodefn:
            if (order < max_idx):
                skip.add(int(fairseqidx))
            else:
                X.add(int(fairseqidx))
                dataset_idx = dataset_idxes[0]
                visible_idx.add(dataset_idx)

    # 识别剩余的无定义token
    for fairseqidx in range(len(task.target_dictionary)):
        if (fairseqidx not in X) and (fairseqidx not in nodefn) and (fairseqidx not in skip):
            nodefn.add(fairseqidx)

    X = sorted(list(X))

    # 保存要替换的词汇表
    with open(os.path.join(pretrained_dir, save_dir, "replaced_vocabs.txt"), "w") as f:
        f.write(json.dumps(X))
    print("saving replaced vocabs to {}".format(os.path.join(pretrained_dir, save_dir, "replaced_vocabs.txt")))

    # 验证分类的完整性
    assert len(X) + len(nodefn) + len(skip) == len(task.target_dictionary)
    print("loading {} token embedding from {} sentences wikidictionary-en, {} tokens has no definition, {} tokens are skipped".format(len(X), len(visible_idx), len(nodefn), len(skip)))

    # ======================== Embedding替换部分 ========================

    if filter_strategy == "task_specific":
        # 简单替换策略：直接复制对应的embeddings
        encoder_emb_weight = model.model.model.encoder.embed_tokens.weight.clone().detach()
        reconstructed_encoder_emb_weight = pretrained_model.model.model.encoder.embed_tokens.weight.clone().detach()
        decoder_emb_weight = model.model.model.decoder.embed_tokens.weight.clone().detach()
        reconstructed_decoder_emb_weight = pretrained_model.model.model.decoder.embed_tokens.weight.clone().detach()
        for vocab in X:
            encoder_emb_weight[vocab] = reconstructed_encoder_emb_weight[vocab]
            decoder_emb_weight[vocab] = reconstructed_decoder_emb_weight[vocab]

    else:
        encoder_emb_weight = load_and_replace_embeddings(model, pretrained_model, wikien_dataset, task, nodefn, skip, visible_idx, embed_type="encoder")
        decoder_emb_weight = load_and_replace_embeddings(model, pretrained_decoder_model, wikien_dataset, task, nodefn, skip, visible_idx, embed_type="decoder")

    # ======================== 更新模型权重 ========================
    state_dict = model.state_dict()
    assert encoder_emb_weight.shape == state_dict['model.model.encoder.embed_tokens.weight'].shape
    assert decoder_emb_weight.shape == state_dict['model.model.decoder.embed_tokens.weight'].shape

    state_dict['model.model.encoder.embed_tokens.weight'] = encoder_emb_weight
    state_dict['model.model.decoder.embed_tokens.weight'] = decoder_emb_weight
    print("loading {} token embedding from {} sentences wikidictionary-en, {} tokens has no definition, {} tokens are skipped".format(len(X), len(visible_idx), len(nodefn), len(skip)))

    #print("weight before loading dictemb", model.model.decoder.embed_tokens.weight[100])
    model.load_state_dict(state_dict)
    #print("weight after loading dictemb", model.model.decoder.embed_tokens.weight[100])

    # print used time
    used_time = datetime.datetime.now() - now
    print("used time: ", used_time)

# ======================== 保存模型（可选） ========================
SAVE_MODEL = True  # 设置为True以保存模型
if SAVE_MODEL:
    checkpoint = torch.load(os.path.join(pretrained_dir, "model.pt"), weights_only=False)
    checkpoint['model'] = model.state_dict()
    torch.save(checkpoint, os.path.join(save_dir, "model.pt"))
