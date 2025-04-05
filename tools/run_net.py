# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""
import os
import sys

sys.path.append(os.getcwd())
from timesformer.utils.misc import launch_job
from timesformer.utils.parser import load_config, parse_args
from tools.train_net import train
import numpy as np
import cl_utils
import dataset_config


def get_func(cfg, args):
    train_func = train
    return train_func


def main():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '5,6'
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = parse_args()
    if args.num_shards > 1:
        args.output_dir = str(args.job_dir)

    cfg = load_config(args)

    # CIL task
    num_class, args.train_list, args.val_list, args.balance_list, args.root_path, prefix = dataset_config.return_dataset(cfg.TRAIN.DATASET)
    args.num_task = int(np.ceil((num_class - args.init_task) / args.nb_class)) + 1
    class_list_total = np.arange(num_class) + 1
    np.random.seed(1000)
    np.random.shuffle(class_list_total)
    total_task_list = class_list_total.tolist()
    total_task_list_save = total_task_list
    total_task_list = cl_utils.set_task(args, total_task_list, num_class)
    end_task = min(args.end_task, args.num_task)

    for i in range(args.start_task, end_task):
        print("----AGE {}----".format(i))
        current_task = total_task_list[i]
        current_head = sum(len(j) for j in total_task_list[:i + 1])
        print('current_task ', current_task)
        print('current_head ', current_head)

        cfg.ROOT_PATH = args.root_path
        cfg.TRAIN_LIST = args.train_list
        cfg.BALANCE_LIST = args.balance_list
        cfg.VAL_LIST = args.val_list
        cfg.CURRENT_TASK = current_task
        cfg.LOSS_WEIGHT = args.casual_loss_weight
        cfg.LOGITS_WEIGHT = args.casual_logits_weight
        cfg.COMPENSATE_EFFECT = args.compensate_effect
        cfg.SUPERVISE_LOSS = args.supervise_loss
        cfg.CASUAL = args.casual
        cfg.ADAPTER = args.adapter
        cfg.HEAD_TYPE = 'linear'
        cfg.DIST = False
        cfg.DIR = args.output_dir
        cfg.PATH_WEIGHT = args.path_weight

        if i > 0:
            cfg.FLAG = 0
        if i == 0:
            cfg.NUM_HEAD = current_head  # num of classifier head when training and testing
        elif i > 0:
            cfg.NUM_HEAD = sum(len(j) for j in total_task_list[:i])
            cfg.NUM_OLD_HEAD = sum(len(j) for j in total_task_list[:i])
        cfg.MODEL.NUM_CLASSES = current_head
        cfg.CLASS_INDEXER = total_task_list_save
        # 直接从task1开始训练，之前task0权重已训练好
        if args.start_task == 0:
            if i == 0:
                cfg.TEST_TASK = total_task_list[0]
            else:
                cfg.EX_TASK = []
                for j in range(0, i):
                    cfg.EX_TASK += total_task_list[j]
                cfg.TEST_TASK = total_task_list[i] + cfg.TEST_TASK
        else:
            if i == args.start_task:
                cfg.TEST_TASK = []
                cfg.EX_TASK = []
                # EX task for balance training
                for j in range(0, i + 1):
                    cfg.TEST_TASK += total_task_list[j]
                for j in range(0, i):
                    cfg.EX_TASK += total_task_list[j]
                print(len(cfg.TEST_TASK)/(args.start_task + 1))
            else:
                cfg.EX_TASK = cfg.TEST_TASK
                cfg.TEST_TASK = total_task_list[i] + cfg.TEST_TASK
        cfg.TEST_TASK = [38]
        cfg.AGE = i
        print(cfg.NUM_HEAD)
        task_len_list = []
        for task in total_task_list:
            length = len(task)
            task_len_list.append(length)
        cfg.LEN_lIST = task_len_list
        train_task = get_func(cfg, args)
        # Perform training.
        if cfg.TRAIN.ENABLE:
            launch_job(cfg=cfg, init_method=args.init_method, args=args, func=train_task)


if __name__ == "__main__":
    main()
