#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import logging
import math
import os
import sys
from tqdm import tqdm #code
from sklearn.decomposition import PCA #code
from typing import Any, Callable, Dict, List, Optional, Tuple

# We need to setup root logger before importing any fairseq libraries.
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")

import numpy as np
import torch
import torch.nn.functional as F #code
from omegaconf import DictConfig, OmegaConf

from fairseq import checkpoint_utils, options, quantization_utils, tasks, utils
from fairseq.data import data_utils, iterators
from fairseq.data.plasma_utils import PlasmaStore
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.initialize import add_defaults
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import fsdp_enable_wrap, fsdp_wrap
from fairseq.distributed import utils as distributed_utils
from fairseq.file_io import PathManager
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer


def main(cfg: FairseqConfig) -> None:
    if isinstance(cfg, argparse.Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)
    add_defaults(cfg)

    if (
        distributed_utils.is_master(cfg.distributed_training)
        and "job_logging_cfg" in cfg
    ):
        # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
        logging.config.dictConfig(OmegaConf.to_container(cfg.job_logging_cfg))

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"
    metrics.reset()

    if cfg.common.log_file is not None:
        handler = logging.FileHandler(filename=cfg.common.log_file)
        logger.addHandler(handler)

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

    # Print args
    logger.info(cfg)
    
    if cfg.checkpoint.write_checkpoints_asynchronously:
        try:
            import iopath  # noqa: F401
        except ImportError:
            logging.exception(
                "Asynchronous checkpoint writing is specified but iopath is "
                "not installed: `pip install iopath`"
            )
            return

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(cfg.task)
    assert cfg.criterion, "Please specify criterion to train a model"

    # Build model and criterion
    if cfg.distributed_training.ddp_backend == "fully_sharded":
        with fsdp_enable_wrap(cfg.distributed_training):
            model = fsdp_wrap(task.build_model(cfg.model))
    else:
        model = task.build_model(cfg.model)

    criterion = task.build_criterion(cfg.criterion)

    #code 20230809
    pretrained_model = None
    if cfg.task['_name'] == 'distill_definition':
        if cfg.model.arch == 'bart_emb_large':
            from fairseq.models.bart import BARTModel
            pretrained_model = BARTModel.from_pretrained(cfg.checkpoint.restore_file, checkpoint_file="model.pt").model

            model_dict = model.state_dict()

            # 1. filter out unnecessary keys
            filtered_dict = {k: v for k, v in pretrained_model.state_dict().items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(filtered_dict)
            # 3. load the new state dict
            model.load_state_dict(model_dict)

            logger.info("copy available parameters from pre-trained model to constructor")

            #freeze parameters
            if not cfg.model.update_emb:
                for params in model.encoder.embed_tokens.parameters():
                    params.requires_grad = False        

        elif cfg.model.arch in ['roberta_emb']:
            from fairseq.models.roberta import RobertaModel
            pretrained_model = RobertaModel.from_pretrained(cfg.checkpoint.restore_file, checkpoint_file="model.pt").model

            model_dict = model.state_dict()

            # 1. filter out unnecessary keys
            filtered_dict = {k: v for k, v in pretrained_model.state_dict().items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(filtered_dict)
            # 3. load the new state dict
            model.load_state_dict(model_dict)

            logger.info("copy available parameters from pre-trained model to constructor")

            #freeze parameters
            if not cfg.model.update_emb:
                for params in model.encoder.sentence_encoder.embed_tokens.parameters():
                    params.requires_grad = False        
        
        elif cfg.model.arch in ["t5_emb_large"]:
            from transformers import T5Model
            pretrained_model = T5Model.from_pretrained("t5-large", cache_dir=cfg.checkpoint.restore_file)

            model_dict = model.model.state_dict()
            # 1. filter out unnecessary keys
            filtered_dict = {k: v for k, v in pretrained_model.state_dict().items() if k in model_dict}
            if len(filtered_dict.keys()) == 0:
                print("No initailized keys from pre-trained models, please check the model architecture")
                exit()

            # 2. overwrite entries in the existing state dict
            model_dict.update(filtered_dict)
            # 3. load the new state dict
            model.model.load_state_dict(model_dict)

            logger.info("copy {} available parameters from pre-trained model to constructor".format(len(filtered_dict.keys())))
            #freeze parameters
            if not cfg.model.update_emb:
                for params in model.model.shared.parameters():
                    params.requires_grad = False  
        
        elif cfg.model.arch in ["t5gemma_emb_l"]:
            from transformers import T5GemmaModel
            pretrained_model = T5GemmaModel.from_pretrained("google/t5gemma-l-l-ul2", cache_dir=cfg.checkpoint.restore_file)

            model_dict = model.model.state_dict()
            # 1. filter out unnecessary keys
            filtered_dict = {k: v for k, v in pretrained_model.state_dict().items() if k in model_dict}
            if len(filtered_dict.keys()) == 0:
                print("No initailized keys from pre-trained models, please check the model architecture")
                exit()

            # 2. overwrite entries in the existing state dict
            model_dict.update(filtered_dict)
            # 3. load the new state dict
            model.model.load_state_dict(model_dict)

            logger.info("copy {} available parameters from pre-trained model to constructor".format(len(filtered_dict.keys())))
            #freeze parameters
            if not cfg.model.update_emb:
                for params in model.model.encoder.embed_tokens.parameters():
                    params.requires_grad = False  
                for params in model.model.decoder.embed_tokens.parameters():
                    params.requires_grad = False

            # t5gemma的encoder和decoder的embedding是分开的
            # decoder和lmhead是绑定在一起的

    if pretrained_model is not None:
        for params in pretrained_model.parameters():
            params.requires_grad = False
        pretrained_model.eval()  # disable dropout (or leave in train mode to finetune)
        
        logger.info(pretrained_model)
        logger.info(
            "num. pre-trained model params: {:,} (num. trained: {:,})".format(
                sum(
                    p.numel() for p in pretrained_model.parameters() if not getattr(p, "expert", False)
                ),
                sum(
                    p.numel()
                    for p in pretrained_model.parameters()
                    if not getattr(p, "expert", False) and p.requires_grad
                ),
            )
        )

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    # We load the valid dataset AFTER building the model
    data_utils.raise_if_valid_subsets_unintentionally_ignored(cfg)
    if cfg.dataset.combine_valid_subsets:
        task.load_dataset("valid", combine=True, epoch=1)
    else:
        for valid_sub_split in cfg.dataset.valid_subset.split(","):
            task.load_dataset(valid_sub_split, combine=False, epoch=1)

    # (optionally) Configure quantization
    if cfg.common.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=cfg.common.quantization_config_path,
            max_epoch=cfg.optimization.max_epoch,
            max_update=cfg.optimization.max_update,
        )
    else:
        quantizer = None

    # Build trainer
    if cfg.common.model_parallel_size == 1:
        trainer = Trainer(cfg, task, model, criterion, quantizer)
    else:
        trainer = MegatronTrainer(cfg, task, model, criterion)

    #code 20230809
    if cfg.model.arch in ['linear'] or cfg.task['_name'] == "distill_definition":
        trainer.build_target_model(pretrained_model)

    logger.info(
        "training on {} devices (GPUs/TPUs)".format(
            cfg.distributed_training.distributed_world_size
        )
    )
    logger.info(
        "max tokens per device = {} and max sentences per device = {}".format(
            cfg.dataset.max_tokens,
            cfg.dataset.batch_size,
        )
    )

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    if cfg.task['_name'] == "distill_definition" and cfg.task.continue_autoregremask:
        cfg.checkpoint.restore_file=cfg.checkpoint.save_dir+"/continue.pt"

    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        cfg.checkpoint,
        trainer,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=task.has_sharded_data("train"),
    )

    if cfg.common.tpu:
        import torch_xla.core.xla_model as xm
        xm.rendezvous("load_checkpoint")  # wait for all workers

    #code
    idx_freq, idx_medium, idx_rare, skip_vocab_left, skip_vocab_right = None, None, None, None, None
    if cfg.checkpoint.save_svd and cfg.criterion["_name"] == "cross_entropy_datamap":
        def load_vocab_index(vocab_path):
            vocab_indexes = []
            with open(vocab_path, encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.split(' ')
                    vocab_indexes.append(int(line[1]))
            return np.array(vocab_indexes)

        idx_freq = load_vocab_index(os.path.join(utils.split_paths(cfg.model.data)[0], 'freq_vocabulary.txt'))
        idx_medium = load_vocab_index(os.path.join(utils.split_paths(cfg.model.data)[0], 'medium_vocabulary.txt'))
        idx_rare = load_vocab_index(os.path.join(utils.split_paths(cfg.model.data)[0], 'rare_vocabulary.txt'))


        skip_vocab_left, skip_vocab_right = 4, None
        if cfg.model.arch == 'hf_t5_large': 
            skip_vocab_left = 3

    # Info model structure
    logger.info(model)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("model: {}".format(model.__class__.__name__))
    logger.info("criterion: {}".format(criterion.__class__.__name__))
    logger.info(
        "num. shared model params: {:,} (num. trained: {:,})".format(
            sum(
                p.numel() for p in model.parameters() if not getattr(p, "expert", False)
            ),
            sum(
                p.numel()
                for p in model.parameters()
                if not getattr(p, "expert", False) and p.requires_grad
            ),
        )
    )
    logger.info(
        "num. expert model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False)),
            sum(
                p.numel()
                for p in model.parameters()
                if getattr(p, "expert", False) and p.requires_grad
            ),
        )
    )

    #code
    if cfg.optimization.do_evaluation:
        valid_losses = validate(cfg, trainer, task, epoch_itr, cfg.dataset.valid_subset.split(","))
        exit()

    if False: #cfg.task.probing:
        if cfg.model.arch == 'bart_large':
            svm_features(cfg, trainer, task, epoch_itr, ['train', 'valid', 'test'])
            exit()
    if cfg.checkpoint.save_svd:
        save_svd_matrix(cfg.checkpoint.save_dir, 0, trainer.get_model(), cfg.model._name, idx_freq, idx_medium, idx_rare, skip_vocab_left, skip_vocab_right)

    max_epoch = cfg.optimization.max_epoch or math.inf
    lr = trainer.get_lr()

    train_meter = meters.StopwatchMeter()
    train_meter.start()
    while epoch_itr.next_epoch_idx <= max_epoch:
        if lr <= cfg.optimization.stop_min_lr:
            logger.info(
                f"stopping training because current learning rate ({lr}) is smaller "
                "than or equal to minimum learning rate "
                f"(--stop-min-lr={cfg.optimization.stop_min_lr})"
            )
            break

        # train for one epoch
        #valid_losses, should_stop = train(cfg, trainer, task, epoch_itr)
        valid_losses, should_stop = train(cfg, trainer, task, epoch_itr, idx_freq, idx_medium, idx_rare, skip_vocab_left, skip_vocab_right) #code
        if should_stop:
            break

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])
        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=task.has_sharded_data("train"),
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=task.has_sharded_data("train"),
        )
        
    #code
    if cfg.checkpoint.save_svd:
        arr = np.column_stack([trainer._criterion.count, trainer._criterion.correctness, trainer._criterion.sum_x, trainer._criterion.sum_xsq])
        datamap_save_dir = os.path.join(cfg.checkpoint.save_dir, 'datamap')
        if not os.path.exists(datamap_save_dir):
            os.makedirs(datamap_save_dir, exist_ok=True)
        np.savetxt(os.path.join(datamap_save_dir, 'count_correctness_sumx_sumxsq.txt'), arr, delimiter=' ')

    train_meter.stop()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))

    # ioPath implementation to wait for all asynchronous file writes to complete.
    if cfg.checkpoint.write_checkpoints_asynchronously:
        logger.info(
            "ioPath PathManager waiting for all asynchronous checkpoint "
            "writes to finish."
        )
        PathManager.async_close()
        logger.info("ioPath PathManager finished waiting.")


def should_stop_early(cfg: DictConfig, valid_loss: float) -> bool:
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if cfg.checkpoint.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if cfg.checkpoint.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= cfg.checkpoint.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    cfg.checkpoint.patience
                )
            )
            return True
        else:
            return False


@metrics.aggregate("train")
def train(
    cfg: DictConfig, trainer: Trainer, task: tasks.FairseqTask, epoch_itr, idx_freq, idx_medium, idx_rare, skip_vocab_left, skip_vocab_right, #code
) -> Tuple[List[Optional[float]], bool]:
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > cfg.dataset.curriculum),
    )
    update_freq = (
        cfg.optimization.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(cfg.optimization.update_freq)
        else cfg.optimization.update_freq[-1]
    )
    itr = iterators.GroupedIterator(
        itr,
        update_freq,
        skip_remainder_batch=cfg.optimization.skip_remainder_batch,
    )
    if cfg.common.tpu:
        itr = utils.tpu_data_loader(itr)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_file=cfg.common.log_file,
        log_interval=cfg.common.log_interval,
        epoch=epoch_itr.epoch,
        aim_repo=(
            cfg.common.aim_repo
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        aim_run_hash=(
            cfg.common.aim_run_hash
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        aim_param_checkpoint_dir=cfg.checkpoint.save_dir,
        tensorboard_logdir=(
            cfg.common.tensorboard_logdir
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        wandb_project=(
            cfg.common.wandb_project
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        wandb_run_name=os.environ.get(
            "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
        ),
        azureml_logging=(
            cfg.common.azureml_logging
            if distributed_utils.is_master(cfg.distributed_training)
            else False
        ),
    )
    progress.update_config(_flatten_config(cfg))

    trainer.begin_epoch(epoch_itr.epoch)

    valid_subsets = cfg.dataset.valid_subset.split(",")
    should_stop = False
    num_updates = trainer.get_num_updates()
    logger.info("Start iterating over samples")

    for i, samples in enumerate(progress):
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
            "train_step-%d" % i
        ):
            log_output = trainer.train_step(samples)

        if log_output is not None:  # not OOM, overflow, ...
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % cfg.common.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()
        valid_losses, should_stop = validate_and_save(
            cfg, trainer, task, epoch_itr, valid_subsets, end_of_epoch, idx_freq, idx_medium, idx_rare, skip_vocab_left, skip_vocab_right, #code
        )

        if should_stop:
            break

    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")
    return valid_losses, should_stop


def _flatten_config(cfg: DictConfig):
    config = OmegaConf.to_container(cfg)
    # remove any legacy Namespaces and replace with a single "args"
    namespace = None
    for k, v in list(config.items()):
        if isinstance(v, argparse.Namespace):
            namespace = v
            del config[k]
    if namespace is not None:
        config["args"] = vars(namespace)
    return config


def validate_and_save(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    valid_subsets: List[str],
    end_of_epoch: bool,
    idx_freq=None, idx_medium=None, idx_rare=None, skip_vocab_left=4, skip_vocab_right=None, #code
) -> Tuple[List[Optional[float]], bool]:
    num_updates = trainer.get_num_updates()
    max_update = cfg.optimization.max_update or math.inf

    # Stopping conditions (and an additional one based on validation loss later
    # on)
    should_stop = False
    if num_updates >= max_update:
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"num_updates: {num_updates} >= max_update: {max_update}"
        )

    training_time_hours = trainer.cumulative_training_time() / (60 * 60)
    if (
        cfg.optimization.stop_time_hours > 0
        and training_time_hours > cfg.optimization.stop_time_hours
    ):
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"cumulative_training_time: {training_time_hours} > "
            f"stop_time_hours: {cfg.optimization.stop_time_hours} hour(s)"
        )

    do_save = (
        (end_of_epoch and epoch_itr.epoch % cfg.checkpoint.save_interval == 0)
        or should_stop
        or (
            cfg.checkpoint.save_interval_updates > 0
            and num_updates > 0
            and num_updates % cfg.checkpoint.save_interval_updates == 0
            and num_updates >= cfg.dataset.validate_after_updates
        )
    )
    #code
    #if num_updates == 3000:
    #    do_save=True
    
    do_validate = (
        (
            (not end_of_epoch and do_save)  # validate during mid-epoch saves
            or (end_of_epoch and epoch_itr.epoch % cfg.dataset.validate_interval == 0)
            or should_stop
            or (
                cfg.dataset.validate_interval_updates > 0
                and num_updates > 0
                and num_updates % cfg.dataset.validate_interval_updates == 0
            )
        )
        and not cfg.dataset.disable_validation
        and num_updates >= cfg.dataset.validate_after_updates
    )

    #code
    do_save_svd = cfg.checkpoint.save_svd and (num_updates % cfg.checkpoint.save_svd_interval_updates == 0 ) and cfg.criterion["_name"] != "cross_entropy_datamap_prompt"
    if do_save_svd:
        save_svd_matrix(cfg.checkpoint.save_dir, num_updates, trainer.get_model(), cfg.model._name, idx_freq, idx_medium, idx_rare, skip_vocab_left, skip_vocab_right)

    # Validate
    valid_losses = [None]
    if do_validate:
        valid_losses = validate(cfg, trainer, task, epoch_itr, valid_subsets)

    should_stop |= should_stop_early(cfg, valid_losses[0])

    # Save checkpoint
    #code temporarly skip
    if do_save or should_stop:
        checkpoint_utils.save_checkpoint(
            cfg.checkpoint, trainer, epoch_itr, valid_losses[0]
        )
        #code
        #if do_save and num_updates == 3000:
        #    exit()

    return valid_losses, should_stop


def get_training_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    subsets: List[str],
) -> List[Optional[float]]:
    """Evaluate the model on the validation set(s) and return the losses."""

    if cfg.dataset.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(cfg.dataset.fixed_validation_seed)

    trainer.begin_valid_epoch(epoch_itr.epoch)
    valid_losses = []
    for subset_idx, subset in enumerate(subsets):
        logger.info('begin validation on "{}" subset'.format(subset))

        # Initialize data iterator
        itr = trainer.get_valid_iterator(subset).next_epoch_itr(
            shuffle=False, set_dataset_epoch=False  # use a fixed valid set
        )
        if cfg.common.tpu:
            itr = utils.tpu_data_loader(itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            aim_repo=(
                cfg.common.aim_repo
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            aim_run_hash=(
                cfg.common.aim_run_hash
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            aim_param_checkpoint_dir=cfg.checkpoint.save_dir,
            tensorboard_logdir=(
                cfg.common.tensorboard_logdir
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
            wandb_project=(
                cfg.common.wandb_project
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            wandb_run_name=os.environ.get(
                "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
            ),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for i, sample in enumerate(progress):
                if (
                    cfg.dataset.max_valid_steps is not None
                    and i > cfg.dataset.max_valid_steps
                ):
                    break

                #code
                #print(task.datasets['valid'].mask_context)
                #mask only one token
                if cfg.model.arch in ['roberta_emb', 'roberta_emb_large'] and cfg.task.mask_context:
                    stack_src_tokens, stack_target, stack_src_lengths, stack_target_masks = [], [], [], []
                    mask_idx = 50264
                    src_item, tgt_item = sample["net_input"]["src_tokens"][0].clone(), sample["target"][0].clone()
                    defn_range = torch.ne(tgt_item[:-1], task.target_dictionary.pad()).nonzero(as_tuple=True)[0]

                    src_defn_tokens, tgt_defn_tokens = tgt_item[defn_range].clone(), src_item[defn_range].clone().fill_(task.target_dictionary.pad())
                    target_mask = sample['target_mask'][0].clone().fill_(False)

                    for position in range(len(defn_range)):
                        _src_defn_tokens, _tgt_defn_tokens = src_defn_tokens.clone(), tgt_defn_tokens.clone()
                        _tgt_defn_tokens[position] = _src_defn_tokens[position].clone()
                        _src_defn_tokens[position] = mask_idx
                        _src_item, _tgt_item = src_item.clone(), tgt_item.clone() 
                        _src_item[defn_range] = _src_defn_tokens
                        _tgt_item[defn_range] = _tgt_defn_tokens
                        _target_mask = target_mask.clone()
                        _target_mask[defn_range[position]] = True

                        stack_src_tokens.append(_src_item)
                        stack_target.append(_tgt_item)
                        stack_src_lengths.append(sample["net_input"]["src_lengths"].clone())
                        stack_target_masks.append(_target_mask)

                    sample["net_input"]["src_tokens"] = torch.stack(stack_src_tokens)
                    sample["target"] = torch.stack(stack_target)
                    sample["target_mask"] = torch.stack(stack_target_masks)
                    sample["net_input"]["src_lengths"] = torch.stack(stack_src_lengths)

                trainer.valid_step(sample)

        # log validation stats
        # only tracking the best metric on the 1st validation subset
        tracking_best = subset_idx == 0
        stats = get_valid_stats(cfg, trainer, agg.get_smoothed_values(), tracking_best)

        if hasattr(task, "post_validate"):
            task.post_validate(trainer.get_model(), stats, agg)

        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[cfg.checkpoint.best_checkpoint_metric])
    return valid_losses

def svm_features(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    subsets: List[str],
) -> List[Optional[float]]:
    """Evaluate the model on the validation set(s) and return the losses."""

    if cfg.dataset.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(cfg.dataset.fixed_validation_seed)

    trainer.begin_valid_epoch(epoch_itr.epoch)
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    with torch.no_grad():
        trainer.model.eval()
        for subset_idx, subset in enumerate(subsets):
            logger.info('begin validation on "{}" subset'.format(subset))

            # Initialize data iterator
            itr = trainer.get_valid_iterator(subset).next_epoch_itr(
                shuffle=False, set_dataset_epoch=False  # use a fixed valid set
            )
            if cfg.common.tpu:
                itr = utils.tpu_data_loader(itr)
            progress = progress_bar.progress_bar(
                itr,
                log_format=cfg.common.log_format,
                log_interval=cfg.common.log_interval,
                epoch=epoch_itr.epoch,
                prefix=f"valid on '{subset}' subset",
                aim_repo=(
                    cfg.common.aim_repo
                    if distributed_utils.is_master(cfg.distributed_training)
                    else None
                ),
                aim_run_hash=(
                    cfg.common.aim_run_hash
                    if distributed_utils.is_master(cfg.distributed_training)
                    else None
                ),
                aim_param_checkpoint_dir=cfg.checkpoint.save_dir,
                tensorboard_logdir=(
                    cfg.common.tensorboard_logdir
                    if distributed_utils.is_master(cfg.distributed_training)
                    else None
                ),
                default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
                wandb_project=(
                    cfg.common.wandb_project
                    if distributed_utils.is_master(cfg.distributed_training)
                    else None
                ),
                wandb_run_name=os.environ.get(
                    "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
                ),
            )

            # create a new root metrics aggregator so validation metrics
            # don't pollute other aggregators (e.g., train meters)
            with metrics.aggregate(new_root=True) as agg:
                for i, sample in enumerate(progress):
                    if (
                        cfg.dataset.max_valid_steps is not None
                        and i > cfg.dataset.max_valid_steps
                    ):
                        break

                    #move to cuda
                    if torch.cuda.is_available():
                        sample = utils.move_to_cuda(sample)

                    src_tokens = sample["net_input"]["src_tokens"]
                    eos = sample["target"].clone().new_full((len(sample["target"]),), task.source_dictionary.eos())
                    eos = eos.unsqueeze(1)  # 将 eos 扩展为二维张量
                    prev_output_tokens = src_tokens[:, :-1].clone()
                    prev_output_tokens = torch.cat((eos, prev_output_tokens), dim=-1)
                    sample["net_input"]["prev_output_tokens"] = prev_output_tokens

                    x, extra = trainer.model(
                        src_tokens=src_tokens,
                        src_lengths=sample["net_input"]["src_lengths"],
                        prev_output_tokens=prev_output_tokens,
                        features_only=True,
                    )
                    sentence_representation = x[src_tokens.eq(eos), :].view(
                        x.size(0), -1, x.size(-1)
                    )[:, -1, :]
                    if subset == 'train':
                        train_X += sentence_representation.cpu().tolist()
                        train_Y += sample["target"].cpu().tolist()
                    else:
                        test_X += sentence_representation.cpu().tolist()
                        test_Y += sample["target"].cpu().tolist()

    # numpy array
    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    test_X = np.array(test_X)
    test_Y = np.array(test_Y)
    print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)

    # train svm
    from sklearn.svm import SVC
    clf = SVC(kernel='linear')
    clf.fit(train_X, train_Y)
    print(clf.score(test_X, test_Y))

    return 

#code
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def save_svd_matrix(save_dir, num_updates, model, arch, idx_freq, idx_medium, idx_rare, skip_vocab_left=4, skip_vocab_right=None):
    svd_save_dir = os.path.join(save_dir, 'svd')
    if not os.path.exists(svd_save_dir):
        os.makedirs(svd_save_dir, exist_ok=True)

    if arch in ['bart_large']:
        word_emb_matrix = model.encoder.embed_tokens.weight[skip_vocab_left:skip_vocab_right].to('cpu').detach().numpy().copy()

    elif arch == 'hf_t5_large':
        word_emb_matrix = model.model.shared.weight[skip_vocab_left:skip_vocab_right].to('cpu').detach().numpy().copy()
        
    elif arch in ['hf_t5gemma_l']:
        # 处理T5Gemma的encoder和decoder embeddings
        encoder_emb_matrix = model.model.model.encoder.embed_tokens.weight[skip_vocab_left:skip_vocab_right].to('cpu').detach().numpy().copy()
        decoder_emb_matrix = model.model.model.decoder.embed_tokens.weight[skip_vocab_left:skip_vocab_right].to('cpu').detach().numpy().copy()
        
        # 对encoder embedding进行SVD
        svd_encoder = TruncatedSVD(n_components=2)
        encoder_emb_truncated = svd_encoder.fit_transform(encoder_emb_matrix)
        np.savetxt(os.path.join(svd_save_dir, f'{num_updates}_encoder.txt'), encoder_emb_truncated, delimiter=',')
        
        # 对decoder embedding进行SVD
        svd_decoder = TruncatedSVD(n_components=2)
        decoder_emb_truncated = svd_decoder.fit_transform(decoder_emb_matrix)
        np.savetxt(os.path.join(svd_save_dir, f'{num_updates}_decoder.txt'), decoder_emb_truncated, delimiter=',')
        
        # 计算unseen词汇的坐标（对encoder和decoder分别处理）
        unseen_x_enc, unseen_y_enc = [], []
        unseen_x_dec, unseen_y_dec = [], []
        for i in range(len(encoder_emb_truncated)):
            if i not in idx_freq and i not in idx_medium and i not in idx_rare:
                unseen_x_enc.append(encoder_emb_truncated[i][0])
                unseen_y_enc.append(encoder_emb_truncated[i][1])
                unseen_x_dec.append(decoder_emb_truncated[i][0])
                unseen_y_dec.append(decoder_emb_truncated[i][1])
        
        unseen_x_enc, unseen_y_enc = np.array(unseen_x_enc), np.array(unseen_y_enc)
        unseen_x_dec, unseen_y_dec = np.array(unseen_x_dec), np.array(unseen_y_dec)
        
        # 画encoder embedding的SVD图
        draw_svd_with_unseen(encoder_emb_truncated, idx_freq, idx_medium, idx_rare, 
                            unseen_x_enc, unseen_y_enc,
                            save_png=os.path.join(svd_save_dir, f'{num_updates}_encoder.png'))
        
        # 画decoder embedding的SVD图
        draw_svd_with_unseen(decoder_emb_truncated, idx_freq, idx_medium, idx_rare,
                            unseen_x_dec, unseen_y_dec,
                            save_png=os.path.join(svd_save_dir, f'{num_updates}_decoder.png'))
        
        return  # 直接返回，因为已经处理完T5Gemma


    # 其他模型的处理保持不变
    svd = TruncatedSVD(n_components=2)
    word_emb_truncated = svd.fit_transform(word_emb_matrix)
    np.savetxt(os.path.join(svd_save_dir, str(num_updates)+'.txt'), word_emb_truncated, delimiter=',')

    unseen_x, unseen_y = [], []
    for i in range(len(word_emb_truncated)):
        if i not in idx_freq and i not in idx_medium and i not in idx_rare:
            unseen_x.append(word_emb_truncated[i][0])
            unseen_y.append(word_emb_truncated[i][1])
    unseen_x, unseen_y = np.array(unseen_x), np.array(unseen_y)

    draw_svd_with_unseen(word_emb_truncated, idx_freq, idx_medium, idx_rare, 
                        unseen_x, unseen_y,
                        save_png=os.path.join(svd_save_dir, str(num_updates)+'.png'))


def draw_svd_with_unseen(matrix_truncated, idx_freq, idx_medium, idx_rare, 
                         unseen_x, unseen_y, save_png='save.png'):
    """独立的draw_svd函数，接受unseen坐标作为参数"""
    freq_x = matrix_truncated[idx_freq, 0]
    freq_y = matrix_truncated[idx_freq, 1]

    medium_x = matrix_truncated[idx_medium, 0]
    medium_y = matrix_truncated[idx_medium, 1]

    rare_x = matrix_truncated[idx_rare, 0]
    rare_y = matrix_truncated[idx_rare, 1]

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)

    plt.tick_params(labelsize=25)

    plt.scatter(freq_x, freq_y,
                marker='o', s=5, color='b',
                label='freq', zorder=1000)

    plt.scatter(medium_x, medium_y,
                marker='x', s=5, color='g',
                label='medium', zorder=1000)

    plt.scatter(rare_x, rare_y,
                marker='^', s=5, color='r',
                label='rare', zorder=1000)

    plt.scatter(unseen_x, unseen_y,
                marker='D', s=5, color='black',
                label='unseen', zorder=1000)

    ax.set_facecolor('0.98')
    plt.grid(alpha=0.6, zorder=1)
    plt.tight_layout()
    plt.savefig(save_png)
    plt.close()

def get_valid_stats(
    cfg: DictConfig,
    trainer: Trainer,
    stats: Dict[str, Any],
    tracking_best: bool,
) -> Dict[str, Any]:
    stats["num_updates"] = trainer.get_num_updates()
    if tracking_best and hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(cfg.checkpoint.best_checkpoint_metric)
        best_function = max if cfg.checkpoint.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[cfg.checkpoint.best_checkpoint_metric],
        )
    return stats


def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    cfg = convert_namespace_to_omegaconf(args)

    if cfg.common.use_plasma_view:
        server = PlasmaStore(path=cfg.common.plasma_path)
        logger.info(
            f"Started plasma server pid {server.server.pid} {cfg.common.plasma_path}"
        )

    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)

    # if cfg.common.use_plasma_view:
    #     server.server.kill()


if __name__ == "__main__":
    cli_main()
