# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Argument parser functions."""

import argparse
import sys

import timesformer.utils.checkpoint as cu
from timesformer.config.defaults import get_cfg


def parse_args():
    """
    Parse the following arguments for a default parser for PySlowFast users.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
    """
    parser = argparse.ArgumentParser(
        description="Provide SlowFast video training and testing pipeline."
    )
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="./configs/HMDB51/TimeSformer_divST_8x32_224.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See slowfast/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument('--nb_class', default=10, type=int, help='class batch')
    parser.add_argument('--init_task', default=11, type=int, help='size of the initial task')
    parser.add_argument('--start_task', default=1, type=int, help='starting task')
    parser.add_argument('--end_task', default=5, type=int, help='last task')
    parser.add_argument('--train_list', type=str,
                        default="./data/hmdb51/hmdb51_train_split_1_rawframes.txt")
    parser.add_argument('--val_list', type=str,
                        default="./data/hmdb51/hmdb51_val_split_1_rawframes.txt/cpfs01/user/liuhuabin/tychen/TCD-main/TCD-main/data/ucf101/ucf101_rgb_val_list.txt")
    parser.add_argument('--root_path', type=str, default="/cpfs01/user/liuhuabin/dataset/HMDB51/jpg")
    parser.add_argument('--casual_loss_weight', default=0.3, type=float, help='casual_loss_weight')
    parser.add_argument('--mix_loss_weight', default=0.2, type=float, help='mix_loss_weight')
    parser.add_argument('--casual_logits_weight', default=0.3, type=float, help='casual_logits_weight')
    parser.add_argument('--casual_enhance', default=False, type=bool, help='whether use casual enhance or not')
    parser.add_argument('--compensate_effect', default=False, type=bool, help='whether use compensate effect or not')
    parser.add_argument('--supervise_loss', default=True, type=bool, help='whether use supervise loss or not')
    parser.add_argument('--casual', default=True, type=bool, help='whether use causal or not')
    parser.add_argument('--adapter', default=True, type=bool, help='whether use adapter or not')
    parser.add_argument('--val_epoch', default=5, type=int, help='from which epoch start validate')
    parser.add_argument('--loss_weight_decay', default=0.03, type=float, help='the decay of loss weight')
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    # Create the checkpoint dir.
    cu.make_checkpoint_dir(cfg.OUTPUT_DIR)
    return cfg
