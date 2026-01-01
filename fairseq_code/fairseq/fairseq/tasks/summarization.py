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
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    prepend_bos=False,
    load_alignments=False,
    truncate_source=False,
    truncate_target=False, #code
    append_source_id=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    prepend_bos_src=None,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    #code
    #for k in itertools.count():
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
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            #code
            if truncate_target:
                tgt_dataset = AppendTokenDataset(
                    TruncateDataset(
                        StripTokenDataset(tgt_dataset, tgt_dict.eos()),
                        256 - 1,
                    ),
                    tgt_dict.eos(),
                )
            tgt_datasets.append(tgt_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        #code
        sample_ratios = [len(src_datasets[0])] * len(src_datasets)
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
    elif prepend_bos_src is not None:
        logger.info(f"prepending src bos: {prepend_bos_src}")
        src_dataset = PrependTokenDataset(src_dataset, prepend_bos_src)

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return LanguagePairDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )


@dataclass
class SummarizationConfig(FairseqDataclass):
    #-------code-----------
    distilling_model_dir: Optional[str] = field(
        default=None, metadata={"help": "the distilling model dir"}
    )
    distilling_model_pt: Optional[str] = field(
        default=None, metadata={"help": "the distilling model pt"}
    )
    load_distilled_embedding: bool = field(
        default=False, metadata={"help": "load distilled embedding"}
    )
    mean_distilled_embedding: bool = field(
        default=False, metadata={"help": "load distilled embedding"}
    )
    remove_pca_component: bool = field(
        default=False, metadata={"help": "load distilled embedding"}
    )
    truncate_target: bool = field(
        default=False, metadata={"help": "truncate target to 256"}
    )
    used_dictionary: bool = field(
        default="bart", metadata={"help": "use bart/t5/t5gemma dictionary"}
    )
    #-------code-----------
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
    load_alignments: bool = field(
        default=False, metadata={"help": "load the binarized alignments"}
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
    upsample_primary: int = field(
        default=-1, metadata={"help": "the amount of upsample primary dataset"}
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


@register_task("summarization", dataclass=SummarizationConfig)
class SummarizationTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    cfg: SummarizationConfig

    def __init__(self, cfg: SummarizationConfig, src_dict, tgt_dict):
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
    def setup_task(cls, cfg: SummarizationConfig, **kwargs):
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
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            #code
            paths = paths[:1]

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
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            load_alignments=self.cfg.load_alignments,
            truncate_source=self.cfg.truncate_source,
            truncate_target=self.cfg.truncate_target, #code
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.cfg.required_seq_len_multiple,
        )

    #code 20230912
    def load_dataset_from_path(self, split, path, epoch=1, shuffle=True, combine=False, **kwargs):
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
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            load_alignments=self.cfg.load_alignments,
            truncate_source=self.cfg.truncate_source,
            truncate_target=self.cfg.truncate_target,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=shuffle, #code
            pad_to_multiple=self.cfg.required_seq_len_multiple,
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

    def build_dataset_fromsrctgt_for_inference(self, src_tokens, tgt_tokens):
        src_lengths = torch.LongTensor([t.numel() for t in src_tokens])
        tgt_lengths = torch.LongTensor([t.numel() for t in tgt_tokens])
        return LanguagePairDataset(
            src_tokens,
            src_lengths,
            self.source_dictionary,
            tgt=tgt_tokens,
            tgt_sizes=tgt_lengths,
            tgt_dict=self.target_dictionary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=False,
            pad_to_multiple=self.cfg.required_seq_len_multiple,
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
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False, 
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
                loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output