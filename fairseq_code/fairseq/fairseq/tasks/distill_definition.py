# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import itertools
import json
import logging
import os
from typing import Optional
from argparse import Namespace
from omegaconf import II

import numpy as np
from fairseq import metrics, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    DefnPairDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
    Dictionary,
    T5Dictionary,
    T5GemmaDictionary,
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask, register_task

#code 20230809
import torch
from fairseq.optim.amp_optimizer import AMPOptimizer
from collections import defaultdict

logger = logging.getLogger(__name__)


def load_langpair_dataset(
    data_paths,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    truncate_source=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    filter_ratio=0,
    filter_strategy="maxidx",
    mask_context=False,
    used_dictionary="bart",
):
    if filter_strategy == "causal_language_modeling":
        left_pad_source = False

    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    #code
    for k in range(len(data_paths)):
        data_path = data_paths[k]
        #code
        split_k = split

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            #code
            if False:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )

        #code
        assert 0 <= filter_ratio < 1
        tgt_masks = []
        if filter_strategy == "maxidx":
            #计算一个阈值索引 max_idx
            #例如：如果词典有10000个token，filter_ratio=0.2，那么 max_idx = 8000
            #这意味着保留词典中前80%的高频token
            max_idx = int((1 - filter_ratio) * len(tgt_dict))
            for x in range(len(tgt_dataset)-1, -1, -1):
                # tokens with tgt_mask=1 will be considered as target embedding
                if used_dictionary=="t5":
                    MASKLABEL = '<extra_id_3>'
                elif used_dictionary in ["bart", "t5gemma"]:
                    MASKLABEL = '<mask>'
                # tgt_mask 为 True 的token需要满足以下任一条件：
                tgt_mask = torch.logical_or(
                    # 条件A（高频有效token）：
                    torch.logical_and(
                        torch.gt(tgt_dataset[x], tgt_dict.pad()), # token索引 > pad索引
                        torch.lt(tgt_dataset[x], max_idx)         # token索引 < max_idx
                    ), 
                    # 条件B（特殊mask token）, 对t5来说， 数据里用到的4个mask label都在字典的最后位，设置本命令来对mask label也进行训练：
                    torch.ge(tgt_dataset[x], tgt_dict.index(MASKLABEL)) # token索引 >= MASKLABEL索引
                )
                if not torch.any(tgt_mask):
                    src_dataset.delete_item(x)
                    tgt_dataset.delete_item(x)
                else:
                    tgt_masks.append(tgt_mask)

        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_datasets.append(tgt_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    src_dataset = src_datasets[0]
    tgt_dataset = tgt_datasets[0] 

    #code
    tgt_masks = tgt_masks[::-1]

    assert len(src_dataset) == len(tgt_dataset) == len(tgt_masks)

    eos = None

    input_feeding = True if filter_strategy != "causal_language_modeling" else False

    tgt_dataset_sizes = tgt_dataset.sizes
    return DefnPairDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        tgt_masks,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
        filter_ratio=filter_ratio,
        mask_context=mask_context,
        input_feeding=input_feeding,
    )


@dataclass
class DistillDefinitionConfig(FairseqDataclass):
    continue_autoregremask: bool = field(
        default=False, metadata={"help": "if set, for model to continue training (due to 24h limit)"}
    )
    filter_ratio: float = field(
        default=0, metadata={"help": "token embedding with least token frequency (percentage) will not be considered as target embedding"}
    )
    filter_ratio_inference: float = field(
        default=0, metadata={"help": "token embedding with least token frequency (percentage) will not be considered as target embedding"}
    )
    All_to_P: bool = field(
        default=False, metadata={"help": "if set, use whole weight matrix to generate principalAxes"}
    )
    mask_context: bool = field(
        default=False, metadata={"help": "if set, mask the input sentence"}
    )
    filter_strategy: str = field(
        default="maxidx", metadata={"help": "the strategy for filtering"}
    )
    used_dictionary: bool = field(
        default="bart", metadata={"help": "use bart/t5/t5gemma dictionary"}
    )

    data: Optional[str] = field(
        default=None,
        metadata={
            "help": "colon separated path to data directories list, will be iterated upon during epochs "
            "in round-robin manner; however, valid and test data are always in the first directory "
            "to avoid the need for repeating them in all directories"
        },
    )
    source_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "source language",
            "argparse_alias": "-s",
        },
    )
    target_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "target language",
            "argparse_alias": "-t",
        },
    )
    left_pad_source: bool = field(
        default=True, metadata={"help": "pad the source on the left"}
    )
    left_pad_target: bool = field(
        default=False, metadata={"help": "pad the target on the left"}
    )
    max_source_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the target sequence"}
    )
    truncate_source: bool = field(
        default=False, metadata={"help": "truncate source to max-source-positions"}
    )
    num_batch_buckets: int = field(
        default=0,
        metadata={
            "help": "if >0, then bucket source and target lengths into "
            "N buckets and pad accordingly; this is useful on TPUs to minimize the number of compilations"
        },
    )
    train_subset: str = II("dataset.train_subset")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    required_seq_len_multiple: int = II("dataset.required_seq_len_multiple")


@register_task("distill_definition", dataclass=DistillDefinitionConfig)
class DistillDefinitionTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    cfg: DistillDefinitionConfig

    def __init__(self, cfg: DistillDefinitionConfig, src_dict, tgt_dict):
        super().__init__(cfg)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def load_dictionary(cls, filename, used_dictionary="bart"):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        if used_dictionary=="bart":
            dictionary = Dictionary.load(filename)
        elif used_dictionary=="t5":
            dictionary = T5Dictionary.load(filename)
        elif used_dictionary=="t5Gemma":
            dictionary = T5GemmaDictionary.load(filename)

        return dictionary

    @classmethod
    def setup_task(cls, cfg: DistillDefinitionConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        # find language pair automatically
        if cfg.source_lang is None or cfg.target_lang is None:
            cfg.source_lang, cfg.target_lang = data_utils.infer_language_pair(paths[0])
        if cfg.source_lang is None or cfg.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.source_lang)), used_dictionary=cfg.used_dictionary
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.target_lang)), used_dictionary=cfg.used_dictionary
        )
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(cfg.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(cfg.target_lang, len(tgt_dict)))

        return cls(cfg, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        #code
        filter_ratio = self.cfg.filter_ratio
        filter_strategy = self.cfg.filter_strategy
        mask_context = self.cfg.mask_context

        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            #code
            paths = paths[:1]
            if self.cfg.filter_ratio == 0 and self.cfg.filter_ratio_inference != 0:
                filter_ratio = self.cfg.filter_ratio_inference
            mask_context = False


        #code
        data_paths = paths

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_paths,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            truncate_source=self.cfg.truncate_source,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.cfg.required_seq_len_multiple,
            filter_ratio=filter_ratio,
            filter_strategy=filter_strategy,
            mask_context=mask_context,
            used_dictionary=self.cfg.used_dictionary,
        )

    #code 20230912
    def load_dataset_from_path(self, split, path, epoch=1, combine=False, filter_strategy="maxidx", **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(path)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            #code
            paths = paths[:1]

        data_paths = paths

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        dataset = load_langpair_dataset(
            data_paths,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            truncate_source=self.cfg.truncate_source,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.cfg.required_seq_len_multiple,
            filter_strategy=filter_strategy, #code
            used_dictionary=self.cfg.used_dictionary,
        )
        return dataset

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        return LanguagePairDataset(
            src_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
        )

    def build_model(self, cfg, from_checkpoint=False):
        model = super().build_model(cfg, from_checkpoint)
        return model

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.cfg.max_source_positions, self.cfg.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    #code 20230809
    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False, target_model=None, P=None
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True
            target_model (~fairseq.models.BaseFairseqModel): code : the target model for compressing
        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            #with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
            with torch.amp.autocast('cuda', enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion([model, target_model, P], sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion, target_model=None, P=None):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion([model, target_model, P], sample)
        return loss, sample_size, logging_output